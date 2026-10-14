import os
import sys
import json
import cv2
import numpy as np

# Ensure MKLDNN flag is set before importing paddle/paddleocr
os.environ["FLAGS_use_mkldnn"] = "0"

def init_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )

def enhance_for_ocr(bgr):
    """
    Apply adaptive CLAHE contrast enhancement and unsharp masking to handle
    blister foil specular glare and low webcam contrast.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Mild unsharp sharpening
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)
    return sharpened

def parse_paddle_results(results):
    items = []
    for res in results:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("rec_boxes", [])
        
        serializable_boxes = []
        for b in boxes:
            if hasattr(b, "tolist"):
                serializable_boxes.append(b.tolist())
            elif isinstance(b, (list, tuple)):
                serializable_boxes.append(list(b))
            else:
                serializable_boxes.append([])

        for i in range(len(texts)):
            t = texts[i]
            s = float(scores[i]) if i < len(scores) else 1.0
            box = serializable_boxes[i] if i < len(serializable_boxes) else []
            clean_t = t.strip()
            if clean_t:
                items.append({
                    "text": clean_t,
                    "confidence": round(s, 4),
                    "box": box
                })
    return items

def run_ocr(ocr, image_path):
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image not found: {image_path}", "items": []}
    
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return {"status": "error", "message": f"Could not read image: {image_path}", "items": []}

        # Combine complementary OCR passes instead of discarding correct short brand names.
        items = []
        seen = set()
        def collect(result):
            for item in parse_paddle_results(result):
                key = "".join(ch for ch in item["text"].upper() if ch.isalnum())
                if len(key) >= 2 and key not in seen:
                    seen.add(key)
                    items.append(item)

        # OCR is CPU-heavy; cap longest dimension while keeping readable print.
        max_dim = 960
        scale = min(1.0, max_dim / max(bgr.shape[:2]))
        ocr_input = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
        collect(ocr.predict(ocr_input))
        # Fast path: do not run three more expensive PaddleOCR passes when
        # readable text was already found. Rotate the strip physically if needed.
        if not items:
            enhanced = enhance_for_ocr(bgr)
            temp_path = image_path + ".enhanced.jpg"
            try:
                cv2.imwrite(temp_path, enhanced)
                collect(ocr.predict(enhanced))
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        return {"status": "ok", "image": image_path, "items": items}
    except Exception as e:
        return {"status": "error", "message": str(e), "items": []}

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # Interactive worker mode via stdin/stdout
        ocr = init_ocr()
        sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
        sys.stdout.flush()
        
        for line in sys.stdin:
            img_path = line.strip()
            if not img_path:
                continue
            if img_path.lower() in ("exit", "quit"):
                break
            result = run_ocr(ocr, img_path)
            sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
            sys.stdout.flush()
    elif len(sys.argv) > 1:
        img_path = sys.argv[1]
        ocr = init_ocr()
        result = run_ocr(ocr, img_path)
        sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
        sys.stdout.flush()
    else:
        print("Usage: ocr_engine.py <image_path> OR ocr_engine.py --worker")

if __name__ == "__main__":
    main()
