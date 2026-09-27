# Centralised config — reads from the project .env file.
# All other modules import from here rather than calling os.getenv directly.

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

LAPTOP_IP  = os.getenv("LAPTOP_IP", "192.168.0.248")
PI_IP      = os.getenv("PI_IP",     "192.168.0.82")
PI_PORT    = int(os.getenv("PI_PORT",         "8001"))
SUPERVISOR_PORT = int(os.getenv("SUPERVISOR_PORT", "8000"))

TAU_INITIAL    = float(os.getenv("TAU_INITIAL",    "0.15"))
FRAME_RATE     = int(os.getenv("FRAME_RATE",       "10"))
BUFFER_SECONDS = int(os.getenv("BUFFER_SECONDS",   "5"))

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "PekingU/rtdetr_r50vd")
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "models/yolov8n.onnx")

DB_PATH   = os.getenv("DB_PATH",   "data/events.db")
CLIPS_DIR = os.getenv("CLIPS_DIR", "clips")
LOGS_DIR  = os.getenv("LOGS_DIR",  "logs")

ROS2_TOPIC  = os.getenv("ROS2_CAMERA_TOPIC", "/warehouse_cam/image_raw")
PI_BASE_URL = f"http://{PI_IP}:{PI_PORT}"
