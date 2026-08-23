"""
gesture_config.py
=================
Central configuration file for the Sign Language Translator.

ALL tunable parameters and constants live here.
No values should be hard-coded in other modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolves correctly regardless of CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Gesture definitions
# Each entry is (label_index, display_name, spoken_word)
# To add a new gesture: append a new tuple and re-collect data + retrain.
# ---------------------------------------------------------------------------
GESTURES: list[dict] = [
    {"id": 0, "name": "Hi",     "spoken": "Hi"},
    {"id": 1, "name": "What",   "spoken": "What"},
    {"id": 2, "name": "Is",     "spoken": "is"},
    {"id": 3, "name": "Your",   "spoken": "your"},
    {"id": 4, "name": "Name",   "spoken": "name"},
    {"id": 5, "name": "My",     "spoken": "My"},
    {"id": 6, "name": "Harry Styles", "spoken": "Harry Styles"},
]

# Derived lookup maps (built from GESTURES — do not edit directly)
GESTURE_NAMES: list[str]       = [g["name"]   for g in GESTURES]
GESTURE_SPOKEN: dict[int, str] = {g["id"]: g["spoken"] for g in GESTURES}
GESTURE_ID_MAP: dict[str, int] = {g["name"]: g["id"]   for g in GESTURES}
NUM_GESTURES: int               = len(GESTURES)

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------
CAMERA_INDEX: int   = 0        # Default webcam (change to 1, 2, … for external)
CAMERA_WIDTH: int   = 640
CAMERA_HEIGHT: int  = 480
CAMERA_FPS: int     = 30       # Requested FPS (actual FPS depends on hardware)

# ---------------------------------------------------------------------------
# MediaPipe Hand Detection
# ---------------------------------------------------------------------------
MAX_NUM_HANDS: int              = 1      # Detect only one hand for stability
DETECTION_CONFIDENCE: float     = 0.70   # Minimum hand-detection confidence
TRACKING_CONFIDENCE: float      = 0.60   # Minimum hand-tracking confidence

# ---------------------------------------------------------------------------
# ML Model / Inference
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float     = 0.80   # Minimum classifier confidence to accept a prediction
MODEL_PATH: Path                = PROJECT_ROOT / "models" / "gesture_model.pkl"

# ---------------------------------------------------------------------------
# Prediction Stabilizer
# ---------------------------------------------------------------------------
STABILITY_WINDOW: int           = 10     # Number of recent frames to keep
STABILITY_MIN_AGREE: int        = 8      # Frames that must agree to confirm a gesture

# ---------------------------------------------------------------------------
# Sentence Builder / Cooldown
# ---------------------------------------------------------------------------
WORD_COOLDOWN_SECONDS: float    = 4.0    # Seconds before the same word can be added again

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Path              = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH: Path       = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------
SAMPLES_PER_GESTURE: int        = 400    # Target samples to collect per gesture

# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------
NUM_LANDMARKS: int              = 21     # MediaPipe hand landmarks
COORDS_PER_LANDMARK: int        = 3      # x, y, z
FEATURE_VECTOR_SIZE: int        = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
UI_WINDOW_TITLE: str            = "AI Sign Language Translator"
UI_THEME: str                   = "dark"      # CustomTkinter theme: "dark" | "light"
UI_COLOR_SCHEME: str            = "blue"      # Accent colour
UI_FONT_FAMILY: str             = "Segoe UI"
UI_UPDATE_MS: int               = 15          # UI camera refresh interval in ms (~66 fps cap)
