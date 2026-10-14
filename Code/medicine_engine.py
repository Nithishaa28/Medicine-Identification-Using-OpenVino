import os
import re
import json
import time
from difflib import SequenceMatcher
import numpy as np
import cv2
import openvino as ov
from ultralytics import YOLO

class OpenVINOMedicineEngine:
    """
    Open-Set Multi-Modal Medicine Identification Engine using OpenVINO.
    Combines:
    1. OpenVINO GPU Visual Feature Embeddings (Metric Retrieval)
    2. OpenVINO GPU YOLO 53-Class Classification Model
    3. PaddleOCR Multi-line Text and Imprint Recognition
    4. Verified Clinical Medicine Reference Database
    """

    def __init__(
        self,
        ref_db_path="medicine_reference/reference_database.json",
        ref_emb_path="medicine_reference/reference_embeddings.npz",
        embedder_xml="models/openvino_embedder/embedder.xml",
        yolo_ov_dir="runs/classify/medicine_runs/medicine_classifier/weights/best_openvino_model",
        preferred_device="GPU"
    ):
        self.ref_db_path = os.path.abspath(ref_db_path)
        self.ref_emb_path = os.path.abspath(ref_emb_path)
        self.embedder_xml = os.path.abspath(embedder_xml)
        self.yolo_ov_dir = os.path.abspath(yolo_ov_dir)

        # 1. Initialize OpenVINO Core
        self.core = ov.Core()
        cache_dir = os.path.abspath("models/ov_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            self.core.set_property({"CACHE_DIR": cache_dir})
        except Exception:
            pass

        available_devices = self.core.available_devices
        if preferred_device.upper() in available_devices:
            self.device = preferred_device.upper()
        elif "GPU" in available_devices:
            self.device = "GPU"
        else:
            self.device = "CPU"
        print(f"[OpenVINO Engine] Using OpenVINO target device: {self.device} (Available: {available_devices})")

        # 2. Load Visual Embedder
        print(f"[OpenVINO Engine] Loading Visual Embedder ({self.embedder_xml})...")
        self.embedder_model = self.core.compile_model(self.embedder_xml, self.device)
        self.embedder_input = self.embedder_model.input(0)
        self.embedder_output = self.embedder_model.output(0)

        # 3. Load YOLO OpenVINO model
        print(f"[OpenVINO Engine] Loading YOLO OpenVINO model ({self.yolo_ov_dir})...")
        self.yolo_model = YOLO(self.yolo_ov_dir, task="classify")

        # Precompute mean ImageNet normalization arrays
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 4. Load Verified Reference Catalog
        with open(self.ref_db_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)
        self.catalog_by_id = {item["id"]: item for item in self.catalog}
        print(f"[OpenVINO Engine] Loaded {len(self.catalog)} verified reference medicines.")

        # 5. Load Precomputed Reference Embeddings
        emb_data = np.load(self.ref_emb_path, allow_pickle=True)
        self.ref_ids = emb_data["ids"]
        self.ref_names = emb_data["names"]
        self.ref_embeddings = emb_data["embeddings"]  # Shape: (N, 576)
        print(f"[OpenVINO Engine] Loaded {len(self.ref_embeddings)} reference visual vectors.")

        # Warmup models so live queries are immediate (< 15 ms)
        try:
            print("[OpenVINO Engine] Warming up OpenVINO models on GPU...")
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            _ = self.extract_visual_embedding(dummy_img)
            _ = self.run_yolo_classification(dummy_img)
            print("[OpenVINO Engine] Warmup complete. Pipeline ready.")
        except Exception as e:
            print(f"[OpenVINO Engine] Warmup note: {e}")

    def extract_visual_embedding(self, image_bgr):
        """
        Extract 576-d L2-normalized visual embedding using OpenVINO.
        Returns: (embedding_vector, latency_ms)
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        arr = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

        t0 = time.perf_counter()
        res = self.embedder_model([tensor])[self.embedder_output]
        t1 = time.perf_counter()

        vec = res[0]
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        latency_ms = (t1 - t0) * 1000.0
        return vec, latency_ms

    def search_visual_database(self, query_vector, top_k=5):
        """
        Perform cosine similarity vector search against reference embeddings.
        """
        similarities = np.dot(self.ref_embeddings, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            med_id = str(self.ref_ids[idx])
            med_name = str(self.ref_names[idx])
            sim = float(similarities[idx])
            candidates.append({
                "id": med_id,
                "name": med_name,
                "similarity": round(sim, 4),
                "item": self.catalog_by_id.get(med_id, {})
            })
        return candidates

    def run_yolo_classification(self, image_bgr):
        """
        Run the 53-class OpenVINO YOLO classification model.
        Returns: (predicted_class_name, confidence, latency_ms, top5_dict)
        """
        t0 = time.perf_counter()
        result = self.yolo_model.predict(
            image_bgr,
            imgsz=224,
            device="intel:gpu" if self.device == "GPU" else "cpu",
            verbose=False
        )[0]
        t1 = time.perf_counter()

        class_id = int(result.probs.top1)
        predicted_name = str(result.names[class_id])
        confidence = float(result.probs.top1conf)
        # Accurate OpenVINO pure inference latency
        latency_ms = float(result.speed.get("inference", (t1 - t0) * 1000.0))

        top5 = {}
        for cid, conf in zip(result.probs.top5, result.probs.top5conf):
            top5[result.names[int(cid)]] = float(conf)

        return predicted_name, confidence, latency_ms, top5

    def match_ocr_text(self, ocr_items):
        """Match visible printed medicine names against ALL catalog entries.

        Handles DOLO-650 / DOLO 650 / DOLO650 and split OCR lines.
        Rejects ingredient-only and strength-only evidence.
        """
        if not ocr_items:
            return None, 0.0, [], []

        def compact(text):
            return re.sub(r"[^A-Z0-9]", "", str(text).upper())

        lines = []
        for item in ocr_items:
            t = str(item.get("text", "")).strip()
            if t and len(t) >= 2 and float(item.get("confidence", 0)) >= 0.25:
                lines.append(t)
        if not lines:
            return None, 0.0, [], []

        # Join neighboring lines for OCR that splits 'DOLO' and '650'.
        sequences = []
        for i in range(len(lines)):
            sequences.append(lines[i])
            for width in (2, 3):
                if i + width <= len(lines):
                    sequences.append(" ".join(lines[i:i + width]))
        # Whole OCR text also supports separated words such as MEFTAL SPAS.
        sequences.append(" ".join(lines))
        normalized = [compact(t) for t in sequences]
        candidates = []
        for entry in self.catalog:
            brand = compact(entry.get("name", ""))
            if len(brand) < 5:
                continue
            aliases = {brand}
            # Only allow aliases closely representing the brand itself.
            for raw in entry.get("ocr_keywords", []):
                alias = compact(raw)
                if len(alias) >= 6 and (alias in brand or brand in alias):
                    aliases.add(alias)
            score = 0.0
            for alias in aliases:
                for text in normalized:
                    if not text:
                        continue
                    # Match with alphanumeric boundaries when possible.
                    if alias == text or (alias in text and len(alias) >= 6):
                        score = max(score, 1.0 if alias == brand else 0.85)
            if score:
                candidates.append((score, len(brand), entry))

        if not candidates:
            return None, 0.0, [], lines
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        winner = candidates[0]
        # Avoid silently choosing between equally plausible distinct names.
        if len(candidates) > 1 and candidates[1][0] == winner[0] and candidates[1][1] == winner[1]:
            return None, 0.0, ["Ambiguous printed medicine names"], lines
        return winner[2], winner[0], ["Printed brand matched: " + winner[2]["name"]], lines

    def analyze_tablet_morphology(self, image_bgr):
        """
        Analyze physical shape and dominant color of loose tablets or blister cavities.
        Returns: {shape, color, aspect_ratio, is_tablet_like}
        """
        h, w = image_bgr.shape[:2]
        # Crop center 70% where tablet is positioned
        ch, cw = int(h * 0.7), int(w * 0.7)
        cy, cx = (h - ch) // 2, (w - cw) // 2
        center_crop = image_bgr[cy:cy+ch, cx:cx+cw]

        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"shape": "Unknown", "color": "Unknown", "aspect_ratio": 1.0, "is_tablet_like": False}

        # Find largest contour
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        if area < 800:
            return {"shape": "Unknown", "color": "Unknown", "aspect_ratio": 1.0, "is_tablet_like": False}

        # Bounding rect & Aspect ratio
        bx, by, bw, bh = cv2.boundingRect(largest_cnt)
        aspect_ratio = round(float(bw) / max(bh, 1), 2)
        perimeter = cv2.arcLength(largest_cnt, True)
        circularity = round(4 * np.pi * (area / (perimeter * perimeter + 1e-6)), 2)

        if circularity >= 0.70 and 0.80 <= aspect_ratio <= 1.25:
            shape_desc = "Round"
        elif 1.25 < aspect_ratio <= 1.85:
            shape_desc = "Oval / Caplet"
        elif aspect_ratio > 1.85:
            shape_desc = "Oblong / Capsule"
        else:
            shape_desc = "Round"

        # Dominant Color in HSV
        mask = np.zeros(center_crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        h_vals = hsv[:, :, 0][mask == 255]
        s_vals = hsv[:, :, 1][mask == 255]
        v_vals = hsv[:, :, 2][mask == 255]

        if len(h_vals) > 0:
            mean_h = float(np.median(h_vals))
            mean_s = float(np.mean(s_vals))
            mean_v = float(np.mean(v_vals))

            # Categorize color
            if mean_s < 45 and mean_v > 130:
                color_desc = "White / Off-White"
            elif (mean_h <= 12 or mean_h >= 160) and mean_s >= 60 and mean_v >= 50:
                color_desc = "Red / Magenta"
            elif 18 <= mean_h <= 36 and mean_s >= 60:
                color_desc = "Yellow"
            elif 12 <= mean_h < 22 and mean_s >= 70:
                color_desc = "Orange"
            elif 38 <= mean_h <= 85 and mean_s >= 45:
                color_desc = "Green"
            elif mean_v < 70:
                color_desc = "Dark / Black"
            elif (mean_h <= 20 or mean_h >= 160) and mean_v < 110:
                color_desc = "Maroon / Brown"
            else:
                color_desc = "Light Color"
        else:
            color_desc = "Unknown"

        return {
            "shape": shape_desc,
            "color": color_desc,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "is_tablet_like": True
        }

    def analyze_medicine(self, image_bgr, ocr_items=None, visual_cache=None):
        """
        End-to-end multi-modal analysis.
        Follows open-set decision criteria:
        - Can match printed text for catalogued medicines absent from YOLO classes.
        - Provides unverified visual candidates when the printed brand is hidden.
        - Describes loose tablet appearance; does not prove identity.
        - Enforces medical safety: unmarked loose tablets yield tentative matches with explicit warnings.
        - Rejects unverified names and returns 'Unknown Medicine' when evidence is insufficient.
        """
        # Reuse GPU results when OCR arrives; never run the same visual model twice.
        if visual_cache is None:
            query_vec, emb_latency = self.extract_visual_embedding(image_bgr)
            visual_matches = self.search_visual_database(query_vec, top_k=5)
            yolo_pred, yolo_conf, yolo_latency, yolo_top5 = self.run_yolo_classification(image_bgr)
            visual_cache = (visual_matches, yolo_pred, yolo_conf, yolo_latency, yolo_top5, emb_latency)
        else:
            visual_matches, yolo_pred, yolo_conf, yolo_latency, yolo_top5, emb_latency = visual_cache
        top_visual = visual_matches[0] if visual_matches else None
        top_visual_sim = top_visual["similarity"] if top_visual else 0.0
        top_visual_name = top_visual["name"] if top_visual else "None"

        # Step 3: OCR Text Matching
        ocr_candidate, ocr_score, ocr_evidence, ocr_lines = self.match_ocr_text(ocr_items)

        # Step 4: Physical Shape and Color Morphology
        morphology = self.analyze_tablet_morphology(image_bgr)

        total_ov_latency = emb_latency + yolo_latency

        # Decision Thresholds
        OCR_HIGH_CONF_THRESHOLD = 0.40
        OCR_MODERATE_THRESHOLD = 0.25
        VISUAL_STRONG_SIM = 0.78
        VISUAL_MODERATE_SIM = 0.72

        identified = False
        final_name = "Unknown Medicine"
        final_strength = "Unverified"
        final_form = "Unspecified Form"
        final_generic = "Not Identified"
        final_therapeutic = "N/A"
        final_status = "UNKNOWN MEDICINE"
        is_untrained = False
        safety_warning = None
        evidence_summary = []

        # A low reference similarity must NOT prevent a trained-model prediction
        # from being displayed as tentative. It is not proof of identity.
        is_adversarial = any("SHUTTERSTOCK" in l.upper() or "WATERMARK" in l.upper() for l in ocr_lines)
        if is_adversarial:
            ocr_candidate = None
            ocr_score = 0.0

        # Decision Pathway 1: Strong OCR match (e.g. Neurobion Forte, Pan 40, Meftal-Spas visible)
        if ocr_candidate is not None and ocr_score >= OCR_HIGH_CONF_THRESHOLD:
            identified = True
            final_name = ocr_candidate["name"]
            final_strength = ocr_candidate["strength"]
            final_form = ocr_candidate["dosage_form"]
            final_generic = ocr_candidate["generic_name"]
            final_therapeutic = ocr_candidate["therapeutic_class"]
            is_untrained = not ocr_candidate.get("is_in_53_classes", False)

            visual_agrees = (top_visual and top_visual["id"] == ocr_candidate["id"])
            if is_untrained:
                if visual_agrees:
                    final_status = "IDENTIFIED (Untrained Class - Verified via OCR & Visual Match)"
                else:
                    final_status = "IDENTIFIED (Untrained Class - Verified via Reference OCR)"
            else:
                if yolo_pred == ocr_candidate.get("yolo_class_name"):
                    final_status = "IDENTIFIED (Verified - YOLO & OCR Agree)"
                else:
                    final_status = "IDENTIFIED (Verified - Package Text Overrides Classifier)"

            evidence_summary.extend(ocr_evidence)
            if visual_agrees:
                evidence_summary.append(f"Visual embedding match ({top_visual_sim:.2f})")

        # Decision Pathway 2: Moderate OCR match supported by visual retrieval or morphology
        elif ocr_candidate is not None and ocr_score >= OCR_MODERATE_THRESHOLD:
            visual_agrees = (top_visual and top_visual["id"] == ocr_candidate["id"])
            yolo_agrees = (ocr_candidate.get("yolo_class_name") == yolo_pred and yolo_conf >= 0.70)

            if visual_agrees or yolo_agrees:
                identified = True
                final_name = ocr_candidate["name"]
                final_strength = ocr_candidate["strength"]
                final_form = ocr_candidate["dosage_form"]
                final_generic = ocr_candidate["generic_name"]
                final_therapeutic = ocr_candidate["therapeutic_class"]
                is_untrained = not ocr_candidate.get("is_in_53_classes", False)
                final_status = "IDENTIFIED (Cross-verified: Text + Visual Features)"
                evidence_summary.extend(ocr_evidence)
                if visual_agrees:
                    evidence_summary.append(f"Visual similarity ({top_visual_sim:.2f})")
                if yolo_agrees:
                    evidence_summary.append(f"YOLO confidence ({yolo_conf*100:.1f}%)")
            else:
                final_name = "Unknown Medicine"
                final_status = "UNKNOWN MEDICINE"
                safety_warning = "Partial text detected but visual features do not confirm reference item."

        # Decision Pathway 3: no catalogued name found in OCR.
        # Keep OCR pathways above unchanged (including untrained medicines).
        # Visual results are suggestions, not verified medicine identities.
        else:
            identified = False
            final_name = "Unknown Medicine"
            final_strength = "Unverified"
            final_form = "Unverified"
            final_generic = "Unidentified"
            final_therapeutic = "N/A"
            final_status = "UNKNOWN MEDICINE"
            safety_warning = "Visual-only result. Confirm the printed package name before use."

            norm = lambda x: re.sub(r"[^A-Z0-9]", "", str(x).upper())
            trained_entries = [e for e in self.catalog
                               if e.get("is_in_53_classes") and
                               norm(e.get("yolo_class_name") or e.get("name")) == norm(yolo_pred)]
            visual_entry = top_visual.get("item", {}) if top_visual else {}
            visual_is_trained = bool(visual_entry.get("is_in_53_classes"))
            visual_name = visual_entry.get("name") if visual_is_trained else None

            # When the classifier and retrieval disagree, don't silently
            # replace a stronger reference match with the wrong class label.
            if (visual_name and top_visual_sim >= VISUAL_MODERATE_SIM and
                    (norm(visual_name) == norm(yolo_pred) or yolo_conf < 0.75)):
                final_name = visual_name
                final_status = "TENTATIVE (Visual Match - Not Verified)"
                evidence_summary.append(f"Trained visual reference suggests {visual_name} ({top_visual_sim:.2f} similarity)")
                if yolo_pred and norm(visual_name) != norm(yolo_pred):
                    evidence_summary.append(f"Classifier disagrees: {yolo_pred} ({yolo_conf*100:.1f}%)")
            elif yolo_pred and yolo_conf >= 0.50 and len(trained_entries) == 1:
                final_name = trained_entries[0]["name"]
                final_status = "TENTATIVE (Trained Model - Not Verified)"
                evidence_summary.append(f"Trained classifier suggests {yolo_pred} ({yolo_conf*100:.1f}%)")
            elif yolo_pred:
                evidence_summary.append(f"Low-confidence classifier suggestion: {yolo_pred} ({yolo_conf*100:.1f}%)")

        return {
            "_visual_cache": visual_cache,
            "identified": identified,
            "name": final_name,
            "strength": final_strength,
            "dosage_form": final_form,
            "generic_name": final_generic,
            "therapeutic_class": final_therapeutic,
            "status": final_status,
            "is_untrained_class": is_untrained,
            "safety_warning": safety_warning,
            "morphology": morphology,
            "evidence": evidence_summary,
            "openvino": {
                "device": self.device,
                "embedder_latency_ms": round(emb_latency, 2),
                "yolo_latency_ms": round(yolo_latency, 2),
                "total_inference_ms": round(total_ov_latency, 2)
            },
            "visual_search": {
                "top_candidate": top_visual_name,
                "similarity": top_visual_sim,
                "all_candidates": visual_matches
            },
            "yolo_classifier": {
                "prediction": yolo_pred,
                "confidence": round(yolo_conf * 100, 2),
                "top5": yolo_top5
            },
            "ocr_text": {
                "lines_detected": ocr_lines,
                "matched_candidate": ocr_candidate["name"] if ocr_candidate else None,
                "score": ocr_score
            }
        }
