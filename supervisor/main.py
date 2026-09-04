# Supervisor — the laptop-side brain of AECS-SDC.
#
# Accepts a WebSocket connection from the Pi, runs the teacher model on each
# incoming frame, compares it against the student score, and decides whether
# to harvest a clip.  Also exposes REST endpoints for the dashboard and for
# receiving uploaded clips from the Pi.

import asyncio
import json
import os
import sqlite3
import sys

import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))

from config import CLIPS_DIR, SUPERVISOR_PORT
from database import Database
from discrepancy import DiscrepancyEngine
from teacher import TeacherModel
import drone_sensors
drone_sensors.start()


def _detect_scenario(frame_bgr: np.ndarray, accel_mag: float) -> str:
    if accel_mag > 0.5:
        return "motion"
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 50:
        return "low_light"
    if cv2.Canny(gray, 50, 150).mean() < 5:
        return "occlusion"
    return "nominal"

os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logger.add("logs/supervisor.log", rotation="10 MB")

db      = Database(os.path.join(os.path.dirname(__file__), "..", "data", "events.db"))
teacher = TeacherModel()
engine  = DiscrepancyEngine(db=db)
engine._last_teacher_conf = 0.0

connected_pi: WebSocket | None = None
_recording = False  # only log to DB when True
_processing = False  # drop frames while previous is still processing

# Latest annotated frames for the /preview endpoints
_latest_student_jpeg: bytes = b""
_latest_teacher_jpeg: bytes = b""


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Supervisor ready")
    yield
    logger.info("Supervisor shutting down")


app = FastAPI(title="AECS-SDC Supervisor", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return HTMLResponse("""
    <html><body style="font-family:monospace;padding:2rem">
    <h2>AECS-SDC Supervisor</h2>
    <ul>
      <li>WS  /telemetry                    — Pi streams frames + scores</li>
      <li>POST /harvest/upload/{frame_id}   — Pi uploads MP4 clip</li>
      <li>PUT  /config                      — update tau / sensitivity</li>
      <li>PUT  /scenario                    — set active test scenario</li>
      <li>GET  /status                      — system health</li>
      <li>GET  /results                     — per-scenario harvest stats</li>
      <li>GET  /events                      — recent event log</li>
    </ul>
    </body></html>
    """)


def _update_previews(frame_bgr, teacher_dets, student_dets):
    global _latest_teacher_jpeg, _latest_student_jpeg
    _latest_teacher_jpeg = _draw_boxes(frame_bgr, teacher_dets, (0, 200, 0))
    _latest_student_jpeg = _draw_boxes(frame_bgr, student_dets, (200, 100, 0))


def _draw_boxes(frame_bgr: np.ndarray, detections: list, color: tuple) -> bytes:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for d in detections:
        x1 = int((d["cx"] - d["bw"] / 2) * w)
        y1 = int((d["cy"] - d["bh"] / 2) * h)
        x2 = int((d["cx"] + d["bw"] / 2) * w)
        y2 = int((d["cy"] + d["bh"] / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.get('name', d.get('label','?'))} {d['score']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


@app.websocket("/telemetry")
async def telemetry_ws(websocket: WebSocket):
    global connected_pi
    await websocket.accept()
    connected_pi = websocket
    logger.info("Pi connected")

    try:
        while True:
            raw          = await websocket.receive_bytes()
            header_bytes = raw[:2048]
            jpeg_bytes   = raw[2048:]

            try:
                header = json.loads(header_bytes.decode().strip("\x00"))
            except Exception:
                logger.warning("Malformed header — skipping frame")
                continue

            frame_id        = header.get("frame_id", "unknown")
            c_student       = float(header.get("c_student", 0.0))
            entropy         = float(header.get("entropy", 0.0))
            student_detections = header.get("detections", [])

            # Read drone sensors directly from laptop USB connection
            _dsensors = drone_sensors.read()
            sonar_m   = _dsensors["sonar_m"]
            accel_mag = _dsensors["accel_mag"]

            frame_bgr = cv2.imdecode(
                np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame_bgr is None:
                logger.warning(f"Could not decode JPEG for {frame_id}")
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            global _processing
            if _processing:
                await websocket.send_json({"frame_id": frame_id, "skipped": True})
                continue
            _processing = True
            try:
                ev_loop = asyncio.get_running_loop()
                # Run teacher every 5th frame only, in a thread so WS never blocks
                if engine.total_frames % 5 == 0:
                    teacher_result = await ev_loop.run_in_executor(
                        None, teacher.infer, frame_rgb
                    )
                    engine._last_teacher_conf       = teacher_result["top_confidence"]
                    engine._last_teacher_detections = teacher_result["detections"]
                    ev_loop.run_in_executor(None, _update_previews, frame_bgr.copy(),
                                            teacher_result["detections"], student_detections[:])
                else:
                    teacher_result = {
                        "top_confidence": engine._last_teacher_conf,
                        "detections":     getattr(engine, "_last_teacher_detections", []),
                    }

                engine.set_scenario(_detect_scenario(frame_bgr, accel_mag))
                result = await engine.evaluate(
                    frame_id, teacher_result["top_confidence"], c_student, entropy,
                    sonar_m, accel_mag, teacher_result["detections"],
                    record=_recording
                )
                await websocket.send_json(result)
            finally:
                _processing = False

    except WebSocketDisconnect:
        connected_pi = None
        logger.info("Pi disconnected")
    except Exception as e:
        connected_pi = None
        logger.error(f"WebSocket error: {e}")


@app.post("/harvest/upload/{frame_id}")
async def receive_clip(frame_id: str, file: UploadFile = File(...)):
    clip_path  = os.path.join(CLIPS_DIR, f"{frame_id}.mp4")
    label_path = os.path.join(CLIPS_DIR, f"{frame_id}.txt")
    with open(clip_path, "wb") as f:
        f.write(await file.read())
    db.update_clip_path(frame_id, clip_path)

    # Write YOLO-format label file from stored teacher detections
    with sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "events.db")) as conn:
        row = conn.execute(
            "SELECT teacher_labels FROM events WHERE frame_id=? ORDER BY id DESC LIMIT 1",
            (frame_id,)
        ).fetchone()
    if row and row[0]:
        detections = json.loads(row[0])
        with open(label_path, "w") as lf:
            for d in detections:
                lf.write(f"{d['label']} {d['cx']} {d['cy']} {d['bw']} {d['bh']}\n")

    logger.success(f"Clip saved: {clip_path} ({len(detections) if row and row[0] else 0} labels)")
    return JSONResponse({"status": "saved", "path": clip_path})


@app.get("/preview/teacher")
async def preview_teacher():
    if not _latest_teacher_jpeg:
        return Response(status_code=204)
    return Response(_latest_teacher_jpeg, media_type="image/jpeg")


@app.get("/preview/student")
async def preview_student():
    if not _latest_student_jpeg:
        return Response(status_code=204)
    return Response(_latest_student_jpeg, media_type="image/jpeg")


@app.put("/recording")
async def set_recording(enabled: bool):
    global _recording
    _recording = enabled
    logger.info(f"Recording {'started' if enabled else 'stopped'}")
    return JSONResponse({"recording": _recording})


@app.put("/config")
async def update_config(tau: float, sensitivity: str = "normal"):
    engine.set_tau(tau)
    db.log_config_change(tau, sensitivity)
    return JSONResponse({"status": "ok", "tau": tau, "sensitivity": sensitivity})


@app.put("/scenario")
async def set_scenario(name: str):
    engine.set_scenario(name)
    return JSONResponse({"status": "ok", "scenario": name})


@app.get("/status")
async def status():
    total = max(engine.total_frames, 1)
    return {
        "pi_connected":     connected_pi is not None,
        "recording":        _recording,
        "harvest_count":    engine.harvest_count,
        "total_frames":     engine.total_frames,
        "harvest_rate_pct": round(engine.harvest_count / total * 100, 2),
        "current_tau":      round(engine.tau, 4),
        "current_scenario": engine.current_scenario,
        "teacher_model":    "RT-DETR-r50vd",
        "student_model":    "YOLOv8n-ONNX",
    }


@app.get("/results")
async def get_results():
    return db.get_results_by_scenario()


@app.get("/events")
async def get_events(n: int = 50):
    return db.get_recent_events(n)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SUPERVISOR_PORT, log_level="info")
