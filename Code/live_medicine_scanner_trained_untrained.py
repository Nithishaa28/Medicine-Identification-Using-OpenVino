import os
import sys
import time
import argparse
import threading
import queue
import cv2
import numpy as np

from medicine_engine import OpenVINOMedicineEngine
from ocr_client import PaddleOCRClient

class LiveMedicineScanner:
    def __init__(self, camera_index=0, preferred_device="GPU", auto_capture=False):
        self.camera_index = camera_index
        self.auto_capture = auto_capture
        self.running = False
        
        print("[1/3] Loading OpenVINO Medicine Engine...")
        self.engine = OpenVINOMedicineEngine(preferred_device=preferred_device)
        
        print("[2/3] Connecting to PaddleOCR Client (ocr_env)...")
        self.ocr_client = PaddleOCRClient()
        
        # Async capture & analysis pipeline
        self.task_queue = queue.Queue(maxsize=1)
        self.is_analyzing = False
        self.analysis_pending = False
        self.last_analysis_result = None
        self.last_ocr_lines = []
        self.last_ocr_error = ""
        self.last_analysis_time = 0
        self.status_message = "Ready. Hold medicine inside the box."
        self.latest_frame = None
        
        # Motion detection for auto-capture
        self.prev_roi_gray = None
        self.steady_counter = 0
        self.STEADY_THRESHOLD = 20  # frames held steady (~0.7s at 30fps)
        self.MOTION_SENSITIVITY = 12.0 # average pixel diff
        self.COOLDOWN_SECONDS = 3.0 # pause before re-triggering auto-capture

    def _analysis_worker(self):
        """Background thread executing OpenVINO + OCR multi-modal analysis."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame_crop, full_frame = task
            self.last_ocr_lines = []
            self.last_ocr_error = ""
            self.is_analyzing = True
            self.status_message = "Analyzing with OpenVINO & PaddleOCR..."

            temp_img_path = "temp_webcam_crop.jpg"
            cv2.imwrite(temp_img_path, frame_crop)

            try:
                started = time.perf_counter()
                # Display the visual model's tentative result immediately.
                # OCR continues in the same background thread, never blocking webcam.
                preliminary = self.engine.analyze_medicine(frame_crop, ocr_items=[])
                self.last_analysis_result = preliminary
                self.status_message = "Visual result ready; reading package text..."

                visual_seconds = time.perf_counter() - started
                ocr_res = self.ocr_client.recognize(temp_img_path, timeout=45)
                if ocr_res.get("status") != "ok":
                    print("[OCR Warning]", ocr_res.get("message", "OCR unavailable"))
                ocr_items = ocr_res.get("items", [])
                self.last_ocr_lines = [str(i.get("text", "")) for i in ocr_items]
                self.last_ocr_error = str(ocr_res.get("message", "")) if ocr_res.get("status") != "ok" else ""

                # Reuse visual predictions: only rerun text matching and decision
                # logic if text is available. (The engine API currently combines
                # the operations, so this does rerun inference when OCR succeeds.)
                if ocr_items:
                    result = self.engine.analyze_medicine(frame_crop, ocr_items=ocr_items, visual_cache=preliminary.get("_visual_cache"))
                else:
                    result = preliminary
                self.last_analysis_result = result
                self.last_analysis_time = time.time()
                elapsed = time.perf_counter() - started
                print(f"[Timing] Visual: {visual_seconds:.2f}s | OCR: {elapsed-visual_seconds:.2f}s | Total: {elapsed:.2f}s")
                self.status_message = f"Scan completed in {elapsed:.1f}s: {result['name']}"
                if ocr_res.get("status") != "ok":
                    self.status_message = "Visual result shown; OCR unavailable: " + self.last_ocr_error[:35]
            except Exception as e:
                self.status_message = f"Analysis Error: {e}"
                print(f"[Scanner Error] {e}")
            finally:
                if os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except Exception:
                        pass
                self.is_analyzing = False
                self.analysis_pending = False
                self.task_queue.task_done()

    def trigger_analysis(self, frame, roi_box):
        """Enqueue the current frame crop for background analysis."""
        if self.is_analyzing or self.analysis_pending:
            self.status_message = "Scan already running; wait for result."
            return False
        if not self.task_queue.empty():
            return False

        rx, ry, rw, rh = roi_box
        crop = frame[ry:ry+rh, rx:rx+rw].copy()
        try:
            self.task_queue.put_nowait((crop, None))
            self.analysis_pending = True
            self.status_message = "Scan queued; analyzing..."
            self.last_analysis_time = time.time()
            return True
        except queue.Full:
            return False

    @staticmethod
    def _draw_clear_panel_text(image, text, origin, font, scale, color, thickness):
        """Antialiased, readable text for the right-hand results panel only."""
        # Preserve original text, placement, colour and line spacing.
        # Duplex has more legible glyphs than SIMPLEX at small screen sizes.
        cv2.putText(image, str(text), origin, cv2.FONT_HERSHEY_DUPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _visible_ocr_name(lines):
        """Display a likely prominent product name from OCR, without model retraining.

        This is a text display heuristic, not pharmaceutical verification.
        """
        import re

        excluded = (
            "mfg", "manufact", "expiry", "exp.", "batch", "mrp", "price",
            "marketed", "address", "licence", "license", "composition",
            "ingredients", "storage", "store in", "keep out", "schedule",
            "warning", "dosage", "directions", "net wt", "customer",
            "email", "www.", "http", "each tablet", "each capsule",
            "contains", "not for", "for external use", "use only",
            "sensitive teeth", "helps reduce", "unverified", "visual",
        )
        manufacturer_words = (
            "reddy", "laborator", "pharma", "healthcare", "limited", "ltd",
            "private", "pvt", "company", "industries", "dr.",
        )
        candidates = []
        expanded_lines = []
        for raw in lines:
            expanded_lines.extend(str(raw).split("|"))
        for index, raw in enumerate(expanded_lines):
            line = " ".join(str(raw).strip().split()).strip("|:;- ")
            low = line.lower()
            if (not 3 <= len(line) <= 42
                    or not re.search(r"[A-Za-z]{3}", line)
                    or any(term in low for term in excluded)
                    or any(term in low for term in manufacturer_words)
                    or re.fullmatch(r"[\d\s.,/%+-]*(?:ml|mg|mcg|g|kg|rs|inr)[\d\s.,/%+-]*", low)):
                continue
            words = line.split()
            if len(words) > 5:
                continue
            # Prioritize a concise, prominent-looking name, not a company or
            # instructions. Prefer alphabetic labels and allow digits in brands.
            letters = sum(c.isalpha() for c in line)
            score = letters / len(line)
            if len(words) <= 3:
                score += 0.2
            if 4 <= len(line) <= 23:
                score += 0.1
            score -= index * 0.015
            candidates.append((score, index, line))

        if not candidates:
            return None
        _, index, name = max(candidates)
        # Adjacent short OCR labels can be part of one printed product name
        # (e.g., 'vantej' followed by 'AQUA').
        if index + 1 < len(expanded_lines):
            next_text = " ".join(str(expanded_lines[index + 1]).strip().split())
            if (2 <= len(next_text) <= 8 and next_text.isupper()
                    and next_text.isalnum()
                    and next_text.lower() not in ("mg", "ml", "mrp", "exp")
                    and len(name) + len(next_text) < 28):
                name += " " + next_text
        return name

    def draw_dashboard(self, frame, roi_box, fps):
        """Render a clean, professional academic HUD overlay."""
        h, w = frame.shape[:2]
        rx, ry, rw, rh = roi_box

        # 1. Draw target scanning reticle
        box_color = (0, 255, 0) if self.auto_capture else (255, 200, 0)
        if self.is_analyzing:
            # Pulsing color while analyzing
            box_color = (0, 165, 255)
        
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
        
        # Corner brackets for target reticle
        bracket_len = 25
        # Top-left
        cv2.line(frame, (rx, ry), (rx + bracket_len, ry), box_color, 4)
        cv2.line(frame, (rx, ry), (rx, ry + bracket_len), box_color, 4)
        # Top-right
        cv2.line(frame, (rx + rw, ry), (rx + rw - bracket_len, ry), box_color, 4)
        cv2.line(frame, (rx + rw, ry), (rx + rw, ry + bracket_len), box_color, 4)
        # Bottom-left
        cv2.line(frame, (rx, ry + rh), (rx + bracket_len, ry + rh), box_color, 4)
        cv2.line(frame, (rx, ry + rh), (rx, ry + rh - bracket_len), box_color, 4)
        # Bottom-right
        cv2.line(frame, (rx + rw, ry + rh), (rx + rw - bracket_len, ry + rh), box_color, 4)
        cv2.line(frame, (rx + rw, ry + rh), (rx + rw, ry + rh - bracket_len), box_color, 4)

        reticle_label = "MEDICINE SCAN TARGET (Place Strip/Tablet Here)"
        cv2.putText(frame, reticle_label, (rx + 10, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        # 2. Side Panel Dashboard (Right side or overlay card)
        panel_w = 470
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        # Dark slate background for panel
        canvas[:, w:] = (28, 28, 30)

        px = w + 15
        py = 30

        # Title
        self._draw_clear_panel_text(canvas, "OPENVINO MEDICINE AI", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
        py += 20
        self._draw_clear_panel_text(canvas, "Open-Set Laptop Webcam Scanner", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        py += 25
        cv2.line(canvas, (px - 5, py), (w + panel_w - 15, py), (60, 60, 65), 1)
        py += 25

        # System Metrics
        ov_dev = self.engine.device
        self._draw_clear_panel_text(canvas, f"Engine: OpenVINO 2026 [{ov_dev}]", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        py += 20
        self._draw_clear_panel_text(canvas, f"Camera FPS: {fps:.1f} | Auto-Scan: {'ON' if self.auto_capture else 'OFF'}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        py += 25
        cv2.line(canvas, (px - 5, py), (w + panel_w - 15, py), (60, 60, 65), 1)
        py += 25

        # Live Analysis Card
        self._draw_clear_panel_text(canvas, "IDENTIFICATION RESULT:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        py += 28

        res = self.last_analysis_result
        if self.is_analyzing:
            self._draw_clear_panel_text(canvas, "[ ANALYZING MEDICINE... ]", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            py += 25
            self._draw_clear_panel_text(canvas, "Extracting OpenVINO vectors & OCR...", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
            py += 35
        elif res is None:
            self._draw_clear_panel_text(canvas, "No medicine scanned yet.", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
            py += 20
            self._draw_clear_panel_text(canvas, "Position medicine in frame and hold steady.", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            py += 35
        else:
            name = res["name"]
            status = res["status"]
            strength = res["strength"]
            dosage_form = res["dosage_form"]
            generic = res["generic_name"]
            is_untrained = res["is_untrained_class"]
            warning = res["safety_warning"]
            ov_latency = res["openvino"]["total_inference_ms"]

            # Color coding based on status
            if "UNKNOWN" in status:
                name_color = (80, 80, 255)  # Red
                status_color = (80, 80, 255)
            elif "TENTATIVE" in status:
                name_color = (0, 180, 255)  # Amber / Orange
                status_color = (0, 180, 255)
            else:
                name_color = (80, 255, 120)  # Green
                status_color = (80, 255, 120)

            # For an unknown class, show readable package text without claiming
            # that the text has been verified as a medicine identity.
            ocr_display_name = None
            if (not res.get("identified", False) or "unknown" in str(name).lower()) and self.last_ocr_lines:
                ocr_display_name = self._visible_ocr_name(self.last_ocr_lines)
            display_name = ocr_display_name or name

            # Medicine Name
            self._draw_clear_panel_text(canvas, display_name[:24], (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, name_color, 2)
            py += 26

            # Status Badge
            badge_text = "[TEXT MATCH - CHECK PACKAGE]"
            if "TENTATIVE" in status:
                badge_text = "[UNVERIFIED MODEL SUGGESTION]"
            elif not res.get("identified", name != "Unknown Medicine"):
                badge_text = "[UNKNOWN MEDICINE]"
            
            if not ocr_display_name:
                self._draw_clear_panel_text(canvas, badge_text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
            py += 25

            # Training-class membership is based on the displayed name, not the
            # classifier's forced top-1 suggestion. OCR names are unverified.
            import re
            from pathlib import Path
            class_dir = Path(__file__).resolve().parent / "dataset" / "train"
            def _class_key(value):
                return re.sub(r"[^a-z0-9]", "", str(value).lower())
            if class_dir.is_dir():
                trained_names = {_class_key(p.name) for p in class_dir.iterdir() if p.is_dir()}
                if _class_key(display_name) in trained_names:
                    training_label = "TRAINING STATUS: TRAINED"
                elif ocr_display_name:
                    training_label = "TRAINING STATUS: UNTRAINED"
                elif not res.get("identified", False) or "TENTATIVE" in str(status):
                    training_label = "TRAINING STATUS: UNKNOWN"
                else:
                    training_label = "TRAINING STATUS: UNTRAINED"
            else:
                training_label = "TRAINING STATUS: UNKNOWN"
            self._draw_clear_panel_text(canvas, training_label, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (120, 235, 235), 1)
            py += 22

            # YOLO trained-class prediction is diagnostic, NOT verified identity.
            classifier = res.get("yolo_classifier") or {}
            prediction = str(classifier.get("prediction") or "Unavailable")
            try:
                prediction_conf = float(classifier.get("confidence", 0.0))
            except (TypeError, ValueError):
                prediction_conf = 0.0
            self._draw_clear_panel_text(canvas, "TRAINED MODEL PREDICTION:", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 215, 255), 1)
            py += 20
            self._draw_clear_panel_text(canvas, f"{prediction[:28]} ({prediction_conf:.1f}%)", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 150), 1)
            py += 20
            self._draw_clear_panel_text(canvas, "Model prediction - NOT verified", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (175, 175, 175), 1)
            py += 25

            # Verified Strength
            self._draw_clear_panel_text(canvas, f"Strength: {strength}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            py += 22

            # Form
            self._draw_clear_panel_text(canvas, f"Form: {dosage_form[:32]}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            py += 22

            # Generic composition
            self._draw_clear_panel_text(canvas, f"Active: {generic[:34]}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
            py += 22
            if len(generic) > 34:
                self._draw_clear_panel_text(canvas, f"        {generic[34:68]}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
                py += 20

            # Physical Morphology (Shape & Color)
            morph = res.get("morphology", {})
            if morph and morph.get("is_tablet_like"):
                self._draw_clear_panel_text(canvas, f"Visual: {morph['shape']} | {morph['color']}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 220, 120), 1)
                py += 22

            # OpenVINO Latency
            py += 5
            self._draw_clear_panel_text(canvas, f"OpenVINO Latency: {ov_latency:.1f} ms", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)
            py += 22

            # Evidence
            if res.get("evidence"):
                ev_str = " + ".join(res["evidence"][:2])
                self._draw_clear_panel_text(canvas, f"Evidence: {ev_str[:38]}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 220, 160), 1)
                py += 22

            # Safety Warning Box
            if warning:
                py += 5
                cv2.rectangle(canvas, (px - 5, py), (w + panel_w - 15, py + 45), (20, 20, 60), -1)
                cv2.rectangle(canvas, (px - 5, py), (w + panel_w - 15, py + 45), (60, 60, 180), 1)
                self._draw_clear_panel_text(canvas, "ADVISORY NOTE:", (px + 5, py + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 255), 1)
                self._draw_clear_panel_text(canvas, warning[:42], (px + 5, py + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
                py += 55

        # OCR evidence is shown explicitly, independently from visual suggestions.
        if self.last_ocr_lines:
            line = "OCR READ: " + " | ".join(self.last_ocr_lines[:3])
            self._draw_clear_panel_text(canvas, line[:65], (px, h - 174),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)
        elif self.last_ocr_error:
            self._draw_clear_panel_text(canvas, ("OCR ERROR: " + self.last_ocr_error)[:65], (px, h - 174),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 140, 255), 1)
        # Always show analysis progress and errors, including OCR delays.
        msg = str(self.status_message)
        self._draw_clear_panel_text(canvas, msg[:57], (px, h - 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 210, 255), 1)

        # Footer / Controls
        py = h - 110
        cv2.line(canvas, (px - 5, py), (w + panel_w - 15, py), (60, 60, 65), 1)
        py += 20
        self._draw_clear_panel_text(canvas, "KEYBOARD CONTROLS:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        py += 18
        self._draw_clear_panel_text(canvas, "[SPACE / C] Instant Capture & Scan", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)
        py += 18
        self._draw_clear_panel_text(canvas, "[A] Toggle Auto-Scan | [R] Reset", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)
        py += 18
        self._draw_clear_panel_text(canvas, "[S] Save Image | [Q / ESC] Quit", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)

        # Draw stability progress bar on camera view
        if self.auto_capture and not self.is_analyzing:
            bar_w = 200
            bar_h = 10
            bx = rx + (rw - bar_w) // 2
            by = ry + rh + 15
            progress = min(self.steady_counter / float(self.STEADY_THRESHOLD), 1.0)
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            cv2.rectangle(canvas, (bx, by), (bx + int(bar_w * progress), by + bar_h), (0, 255, 0), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (180, 180, 180), 1)
            self._draw_clear_panel_text(canvas, "Hold Steady for Auto-Scan", (bx + 15, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        return canvas

    def run(self):
        """Main camera scanning loop."""
        print(f"\n[Scanner] Opening laptop camera (Index {self.camera_index})...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"[Error] Failed to open webcam at index {self.camera_index}.")
            print("Please ensure your camera is connected and not used by another application.")
            return False

        # Set 1280x720 resolution for high-definition strip text and tablet feature capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        worker_thread.start()

        window_name = "Medicine Identification System Using OpenVINO"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)

        print("\nScanner started successfully!")
        print("Show a medicine strip, tablet, bottle, or cream to the camera.")
        print("Press 'Space' or 'C' to scan manually. Press 'Q' to quit.\n")

        fps_timer = time.time()
        fps_counter = 0
        current_fps = 0.0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[Warning] Empty frame from camera.")
                    time.sleep(0.05)
                    continue

                self.latest_frame = frame.copy()
                h, w = frame.shape[:2]
                # Center ROI box (e.g., 300x300 centered in 640x480)
                rw = int(w * 0.78)
                rh = int(h * 0.82)
                rx = (w - rw) // 2
                ry = (h - rh) // 2
                roi_box = (rx, ry, rw, rh)

                # Motion detection within the ROI
                roi_crop = frame[ry:ry+rh, rx:rx+rw]
                roi_gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.GaussianBlur(roi_gray, (21, 21), 0)

                if self.prev_roi_gray is not None and self.auto_capture and not self.is_analyzing:
                    diff = cv2.absdiff(roi_gray, self.prev_roi_gray)
                    mean_diff = np.mean(diff)
                    # Check if an object is present (variance > 10) and steady (mean_diff < 8)
                    roi_var = np.var(roi_crop)
                    now = time.time()
                    cooldown_ok = (now - self.last_analysis_time) > self.COOLDOWN_SECONDS

                    if roi_var > 400.0 and mean_diff < self.MOTION_SENSITIVITY and cooldown_ok:
                        self.steady_counter += 1
                        if self.steady_counter >= self.STEADY_THRESHOLD:
                            # Trigger auto-capture!
                            self.trigger_analysis(frame, roi_box)
                            self.steady_counter = 0
                    else:
                        self.steady_counter = max(0, self.steady_counter - 1)

                self.prev_roi_gray = roi_gray

                # FPS calculation
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()

                # Render UI
                dashboard = self.draw_dashboard(frame, roi_box, current_fps)
                cv2.imshow(window_name, dashboard)

                # Handle Keyboard Input
                key = cv2.waitKeyEx(10) & 0xFF
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Scanner window closed.")
                    break
                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    print("Exiting scanner...")
                    break
                elif key in (ord(' '), ord('c'), ord('C')):  # Space or C: manual scan
                    print("[Manual Trigger] Capturing and analyzing medicine...")
                    if not self.trigger_analysis(frame, roi_box):
                        print("[Scanner] Busy; waiting for current analysis.")
                elif key in (ord('a'), ord('A')):  # Toggle auto-scan
                    self.auto_capture = not self.auto_capture
                    print(f"Auto-scan toggled: {'ON' if self.auto_capture else 'OFF'}")
                elif key in (ord('r'), ord('R')):  # Reset
                    self.last_analysis_result = None
                    self.last_ocr_lines = []
                    self.last_ocr_error = ""
                    self.steady_counter = 0
                    print("Results cleared.")
                elif key in (ord('s'), ord('S')):  # Save image
                    save_name = f"captured_medicine_{int(time.time())}.jpg"
                    rx, ry, rw, rh = roi_box
                    crop_name = save_name.replace("captured_medicine_", "captured_crop_")
                    ok1 = cv2.imwrite(save_name, self.latest_frame if self.latest_frame is not None else frame)
                    ok2 = cv2.imwrite(crop_name, frame[ry:ry+rh, rx:rx+rw])
                    self.status_message = f"Saved: {save_name}" if ok1 and ok2 else "Image save failed"
                    print(f"Saved full frame: {save_name}; cropped medicine: {crop_name}")

        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            # Do not block window shutdown while OCR is still busy.
            if not worker_thread.is_alive() or not self.is_analyzing:
                try:
                    self.ocr_client.close()
                except Exception as exc:
                    print(f"[Scanner] OCR cleanup warning: {exc}")
            else:
                print("[Scanner] OCR worker is busy; closing UI without waiting.")
            print("Scanner shutdown cleanly.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Medicine Identification System Using OpenVINO")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--device", type=str, default="GPU", help="OpenVINO inference device: GPU or CPU (default GPU)")
    parser.add_argument("--no-auto", action="store_true", help="Disable auto-capture (default)")
    parser.add_argument("--auto", action="store_true", help="Enable automatic scanning")
    parser.add_argument("--image", type=str, default=None, help="Scan a static image file instead of live camera")
    args = parser.parse_args()

    if args.image:
        # Static image test mode
        if not os.path.exists(args.image):
            print(f"File not found: {args.image}")
            return
        print(f"Analyzing static image: {args.image}...")
        engine = OpenVINOMedicineEngine(preferred_device=args.device)
        ocr = PaddleOCRClient()
        ocr_res = ocr.recognize(args.image)
        bgr = cv2.imread(args.image)
        res = engine.analyze_medicine(bgr, ocr_items=ocr_res.get("items", []))
        ocr.close()
        print("\n--- RESULTS ---")
        print(f"Medicine Name: {res['name']}")
        print(f"Strength:      {res['strength']}")
        print(f"Dosage Form:   {res['dosage_form']}")
        print(f"Generic:       {res['generic_name']}")
        print(f"Status:        {res['status']}")
        classifier = res.get("yolo_classifier") or {}
        print(f"Trained Model: {classifier.get('prediction', 'Unavailable')}")
        print(f"Confidence:    {classifier.get('confidence', 0)}% (unverified prediction)")
        print(f"Untrained:     {res['is_untrained_class']}")
        print(f"OpenVINO ms:   {res['openvino']['total_inference_ms']} ms ({res['openvino']['device']})")
        if res['safety_warning']:
            print(f"Advisory:      {res['safety_warning']}")
    else:
        # Live camera mode
        scanner = LiveMedicineScanner(
            camera_index=args.camera,
            preferred_device=args.device,
            auto_capture=args.auto and not args.no_auto
        )
        scanner.run()

if __name__ == "__main__":
    main()
