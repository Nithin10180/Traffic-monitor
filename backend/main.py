"""
Traffic Monitoring System — FastAPI Backend
============================================
Run:  uvicorn main:app --reload --port 8000
"""

import io, os, json, base64, time, logging, datetime, pathlib
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.35
LOG_FILE   = "traffic_log.json"
HOST       = "0.0.0.0"
PORT       = 8000

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}

BOX_COLORS = {
    "car":       (0,   229, 255),
    "bus":       (77,  109, 255),
    "truck":     (245, 158,  11),
    "motorbike": (239,  68,  68),
    "bicycle":   (167, 139, 250),
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("traffic")

# ─── DETECTOR ─────────────────────────────────────────────────────────────────
class TrafficDetector:
    def __init__(self):
        logger.info("Loading YOLOv8 model …")
        try:
            import torch
            from ultralytics import YOLO

            # PyTorch 2.6+ defaults weights_only=True which blocks
            # loading ultralytics models. Patch torch.load temporarily.
            _original_load = torch.load
            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _original_load(*args, **kwargs)
            torch.load = _patched_load

            self.model = YOLO(MODEL_PATH)

            # Restore original torch.load
            torch.load = _original_load

            try:
                self.model.fuse()
            except Exception:
                logger.warning("model.fuse() skipped (non-critical)")
            self.ready = True
            logger.info("✓ YOLOv8 model ready")
        except Exception as exc:
            logger.error(f"Could not load YOLOv8: {exc}")
            self.ready = False

    def detect(self, pil_image: Image.Image) -> dict:
        if not self.ready:
            raise RuntimeError("YOLOv8 model not loaded. Run: pip install ultralytics")

        t0 = time.perf_counter()
        results = self.model(pil_image, conf=CONFIDENCE, verbose=False)
        elapsed = time.perf_counter() - t0
        fps = round(1.0 / elapsed) if elapsed > 0 else 0

        counts      = {v: 0 for v in VEHICLE_CLASSES.values()}
        boxes       = []
        confidences = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label = VEHICLE_CLASSES[cls_id]
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            counts[label] += 1
            confidences.append(conf)
            boxes.append({
                "label":      label,
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
            })

        total     = sum(counts.values())
        mean_conf = round(float(np.mean(confidences)), 3) if confidences else 0.0
        status    = self._classify(total)
        annotated = self._draw_boxes(pil_image.copy(), boxes)

        result = {
            "vehicles":        counts,
            "total":           total,
            "traffic_status":  status,
            "confidence":      mean_conf,
            "fps_estimate":    fps,
            "inference_ms":    round(elapsed * 1000, 1),
            "bounding_boxes":  boxes,
            "annotated_image": annotated,
            "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
        }
        self._log(result)
        return result

    @staticmethod
    def _classify(total: int) -> str:
        if total <= 5:    return "Low"
        elif total <= 15: return "Moderate"
        return "High"

    @staticmethod
    def _draw_boxes(image: Image.Image, boxes: list) -> str:
        draw = ImageDraw.Draw(image)
        W, H = image.size
        lw   = max(2, int(min(W, H) / 300))
        fs   = max(14, int(min(W, H) / 40))

        ICONS = {
            "car": "CAR", "bus": "BUS", "truck": "TRUCK",
            "motorbike": "MOTO", "bicycle": "BIKE"
        }

        try:
            font      = ImageFont.truetype("arial.ttf", fs)
            font_bold = ImageFont.truetype("arialbd.ttf", fs)
        except Exception:
            font      = ImageFont.load_default()
            font_bold = font

        for box in boxes:
            label = box["label"]
            conf  = box["confidence"]
            x1, y1, x2, y2 = box["bbox"]
            color = BOX_COLORS.get(label, (0, 229, 255))

            # Main bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

            # Corner accents
            cs = min((x2 - x1), (y2 - y1)) * 0.08
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                draw.rectangle([cx - lw, cy - lw, cx + lw, cy + lw], fill=color)

            # Label
            icon  = ICONS.get(label, label.upper())
            text  = f" {icon}  {conf:.0%} "
            bbox_text = draw.textbbox((0, 0), text, font=font_bold)
            tw    = bbox_text[2] - bbox_text[0]
            th    = bbox_text[3] - bbox_text[1] + 8
            ty    = max(0, y1 - th)

            draw.rectangle([x1, ty, x1 + tw + 4, ty + th], fill=color)
            draw.text((x1 + 4, ty + 4), text, fill=(7, 9, 15), font=font_bold)

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _log(result: dict):
        try:
            log = []
            if pathlib.Path(LOG_FILE).exists():
                with open(LOG_FILE) as f:
                    log = json.load(f)
            log.append({k: result[k] for k in
                        ("timestamp", "total", "traffic_status", "confidence", "vehicles")})
            with open(LOG_FILE, "w") as f:
                json.dump(log[-500:], f, indent=2)
        except Exception:
            pass


# ─── APP ──────────────────────────────────────────────────────────────────────
app      = FastAPI(title="Traffic Monitoring API", version="2.0.0")
detector = TrafficDetector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_ready": detector.ready,
        "model":       MODEL_PATH,
        "version":     "2.0.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG, PNG, or WEBP."
        )
    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="YOLOv8 model not loaded. Run: pip install ultralytics"
        )
    raw    = await file.read()
    image  = Image.open(io.BytesIO(raw)).convert("RGB")
    result = detector.detect(image)
    logger.info(
        f"[/predict] {file.filename}  total={result['total']}  "
        f"status={result['traffic_status']}  {result['inference_ms']}ms"
    )
    return JSONResponse(result)


@app.get("/history")
def history(limit: int = 50):
    if not pathlib.Path(LOG_FILE).exists():
        return []
    with open(LOG_FILE) as f:
        log = json.load(f)
    return log[-limit:]


@app.get("/stats")
def stats():
    if not pathlib.Path(LOG_FILE).exists():
        return {"error": "No log data yet"}
    with open(LOG_FILE) as f:
        log = json.load(f)
    if not log:
        return {}
    totals   = [e["total"] for e in log]
    statuses = [e["traffic_status"] for e in log]
    return {
        "total_detections": len(log),
        "avg_vehicles":     round(sum(totals) / len(totals), 1),
        "max_vehicles":     max(totals),
        "status_counts": {
            "Low":      statuses.count("Low"),
            "Moderate": statuses.count("Moderate"),
            "High":     statuses.count("High"),
        },
        "last_updated": log[-1]["timestamp"],
    }


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║     Traffic Monitoring System  —  Backend v2     ║
╠══════════════════════════════════════════════════╣
║  API  →  http://localhost:8000                   ║
║  Docs →  http://localhost:8000/docs              ║
║  Stop →  Ctrl+C                                  ║
╚══════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
