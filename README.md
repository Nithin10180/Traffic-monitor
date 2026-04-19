# 🚦 TrafficLens — AI Vehicle Detection System

Detects vehicles by category (car, bus, truck, motorbike, bicycle) using YOLOv8 + FastAPI backend and a browser-based frontend.

---

## 📁 Project Structure

```
traffic_system/
├── backend/
│   ├── main.py            ← FastAPI server (YOLOv8 detection)
│   └── requirements.txt   ← Python dependencies
├── frontend/
│   └── index.html         ← Browser UI (open directly)
└── README.md
```

---

## ⚙️ Setup & Run

### Step 1 — Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

> Requires Python 3.9+

---

### Step 2 — Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

On first run, YOLOv8 will automatically download `yolov8n.pt` (~6 MB).

You should see:
```
✓ YOLOv8 model ready
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 3 — Open the Frontend

Simply open `frontend/index.html` in your browser.

> **Tip:** Use VS Code Live Server extension for best experience.
> Right-click `index.html` → Open with Live Server

---

## 🎯 How to Use

1. The status pill top-right will show **YOLOv8 Ready** when backend is connected
2. Upload a traffic image (drag & drop or click)
3. Click **Detect Vehicles**
4. Results show:
   - Annotated image with colored bounding boxes per category
   - Vehicle counts breakdown (car / bus / truck / motorbike / bicycle)
   - Traffic status: **Low** / **Moderate** / **High**
   - Confidence score and inference time
   - Full detections table with bounding box coordinates
5. Switch to **History** tab to see past detections

---

## 🚗 Vehicle Categories & Colors

| Category   | Color  |
|------------|--------|
| 🚗 Car      | Cyan   |
| 🚌 Bus      | Blue   |
| 🚛 Truck    | Amber  |
| 🏍️ Motorbike| Red    |
| 🚲 Bicycle  | Purple |

---

## 🔌 API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | /health    | Backend & model status             |
| POST   | /predict   | Upload image → get detections      |
| GET    | /history   | Past 50 detections log             |
| GET    | /stats     | Aggregate statistics               |
| GET    | /docs      | Interactive Swagger API docs       |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| "Backend Offline" in UI | Make sure `uvicorn` is running on port 8000 |
| "Model Not Loaded" | Run `pip install ultralytics` |
| Slow first detection | YOLOv8 is warming up — subsequent detections are fast |
| CORS error in browser | Backend already allows all origins, refresh the page |
