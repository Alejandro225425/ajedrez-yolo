"""Configuracion central del proyecto."""

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(os.getenv("DATASET_DIR", str(PROJECT_ROOT / "dataset"))).expanduser()
DATA_YAML = Path(os.getenv("DATA_YAML", str(PROJECT_ROOT / "data.yaml"))).expanduser()
RUNS_DIR = PROJECT_ROOT / "runs"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_BEST = MODEL_DIR / "best.pt"
BASE_MODEL = os.getenv("YOLO_BASE_MODEL", "yolov8n.pt")
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("YOLO_EPOCHS", "60"))
PHASE_1_EPOCHS = int(os.getenv("YOLO_PHASE_1_EPOCHS", "30"))
PATIENCE = int(os.getenv("YOLO_PATIENCE", "20"))
CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.40"))
IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))

ESP32_IP = os.getenv("ESP32_CAM_IP", "192.168.4.1")
ESP32_CAPTURE_URL = os.getenv("ESP32_CAPTURE_URL", f"http://{ESP32_IP}/capture")
ESP32_TIMEOUT_SECONDS = float(os.getenv("ESP32_TIMEOUT_SECONDS", "5"))
ESP32_RETRIES = int(os.getenv("ESP32_RETRIES", "3"))

STOCKFISH_PATH = Path(os.getenv(
    "STOCKFISH_PATH",
    str(PROJECT_ROOT / "tools" / "stockfish-windows-x86-64-avx2.exe"),
))
STOCKFISH_TIME_SECONDS = float(os.getenv("STOCKFISH_TIME_SECONDS", "0.5"))

CORNER_JSON = PROJECT_ROOT / "board_corners.json"
WARP_SIZE = int(os.getenv("BOARD_WARP_SIZE", "800"))
