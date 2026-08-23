"""
collect_data.py
===============
Interactive dataset collection script.

Usage:
    python training/collect_data.py

Controls:
    0-9     : Select gesture by number
    SPACE   : Start / pause collection for selected gesture
    ESC/Q   : Quit

The script opens the webcam, shows the live feed with landmarks, and
saves landmark feature vectors to CSV files in data/raw/.

Each CSV row format:
    gesture_id, x0, y0, z0, x1, y1, z1, ... x20, y20, z20
    (1 label column + 63 feature columns = 64 columns total)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera        import Camera
from src.feature_extractor import FeatureExtractor
from src.gesture_config import (
    GESTURES,
    RAW_DATA_DIR,
    SAMPLES_PER_GESTURE,
    NUM_LANDMARKS,
)
from src.hand_detector import HandDetector
from src.utils         import ensure_dir, get_logger

logger = get_logger("collect_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_csv_path(gesture_id: int, gesture_name: str) -> Path:
    """Return the CSV path for a given gesture."""
    safe_name = gesture_name.replace(" ", "_").lower()
    return RAW_DATA_DIR / f"{gesture_id:02d}_{safe_name}.csv"


def count_existing_samples(csv_path: Path) -> int:
    """Count rows already written to a CSV (ignores header if present)."""
    if not csv_path.is_file():
        return 0
    with open(csv_path, "r", newline="") as f:
        return sum(1 for _ in f)


def write_sample(csv_path: Path, gesture_id: int, feature_vector) -> None:
    """Append one sample row to the gesture CSV file."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesture_id] + feature_vector.tolist())


def draw_ui(
    frame,
    selected: dict | None,
    collecting: bool,
    sample_count: int,
    total_target: int,
) -> None:
    """Render collection instructions onto the frame."""
    h, w = frame.shape[:2]

    # Background bar at top
    cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)

    if selected:
        gesture_text = f"Gesture: {selected['name']}  [{selected['id']}]"
    else:
        gesture_text = "No gesture selected — press 0-9"

    cv2.putText(frame, gesture_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    status_color = (0, 220, 0) if collecting else (0, 160, 255)
    status_text  = f"{'COLLECTING' if collecting else 'PAUSED'}  {sample_count}/{total_target}"
    cv2.putText(frame, status_text, (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    # Key guide at bottom
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
    guide = "0-9/N/P: Select | SPACE: Start/Pause | Q: Quit"
    cv2.putText(frame, guide, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    # Gesture list overlay on the right
    x_offset = w - 220
    cv2.rectangle(frame, (x_offset - 5, 0), (w, len(GESTURES) * 24 + 10), (20, 20, 20), -1)
    for i, g in enumerate(GESTURES):
        path = get_csv_path(g["id"], g["name"])
        count = count_existing_samples(path)
        done = count >= total_target
        color = (0, 200, 0) if done else (200, 200, 200)
        marker = "✓" if done else f"{count:3d}"
        label = f"{i}: {g['name'][:10]:<10} {marker}"
        cv2.putText(frame, label, (x_offset, 20 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dir(RAW_DATA_DIR)

    camera   = Camera()
    detector = HandDetector()
    extractor = FeatureExtractor()

    print("\n" + "="*60)
    print(" Sign Language Dataset Collection Tool")
    print("="*60)
    for g in GESTURES:
        path    = get_csv_path(g["id"], g["name"])
        samples = count_existing_samples(path)
        print(f"  [{g['id']}] {g['name']:<12} — {samples}/{SAMPLES_PER_GESTURE} samples")
    print("="*60)
    print(" Press number keys (0-9) to select a gesture.")
    print(" Press SPACE to start/pause collection.")
    print(" Press Q or ESC to quit.\n")

    try:
        camera.start()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    selected_gesture: dict | None = None
    collecting = False
    sample_count = 0

    # Minimum interval between samples (avoid collecting duplicates at 30fps)
    SAMPLE_INTERVAL_S = 0.05  # ~20 samples/sec max
    last_sample_time = 0.0

    cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Data Collection", 800, 600)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            hand = detector.detect(frame)
            if hand:
                frame = detector.draw(frame, hand)

            sample_count_display = (
                count_existing_samples(
                    get_csv_path(selected_gesture["id"], selected_gesture["name"])
                )
                if selected_gesture else 0
            )

            draw_ui(frame, selected_gesture, collecting,
                    sample_count_display, SAMPLES_PER_GESTURE)

            # Collect sample if active
            if collecting and selected_gesture and hand:
                now = time.monotonic()
                if now - last_sample_time >= SAMPLE_INTERVAL_S:
                    fv = extractor.extract(hand.landmarks)
                    if fv is not None:
                        csv_path = get_csv_path(
                            selected_gesture["id"], selected_gesture["name"]
                        )
                        write_sample(csv_path, selected_gesture["id"], fv)
                        last_sample_time = now
                        current_count = count_existing_samples(csv_path)
                        if current_count >= SAMPLES_PER_GESTURE:
                            print(f"  ✓ Target reached for '{selected_gesture['name']}'!")
                            collecting = False

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                print("\nQuitting collection tool.")
                break

            elif key == ord(" "):
                if selected_gesture is None:
                    print("  [!] Select a gesture first (press 0-9).")
                else:
                    collecting = not collecting
                    state = "STARTED" if collecting else "PAUSED"
                    print(f"  Collection {state} for '{selected_gesture['name']}'")

            elif chr(key) in "0123456789":
                idx = int(chr(key))
                if idx < len(GESTURES):
                    selected_gesture = GESTURES[idx]
                    collecting = False
                    print(f"  Selected: [{idx}] {selected_gesture['name']}")
                else:
                    print(f"  [!] No gesture at index {idx}.")
                    
            elif chr(key).lower() == 'n':
                if selected_gesture is None:
                    selected_gesture = GESTURES[0]
                else:
                    selected_gesture = GESTURES[(selected_gesture["id"] + 1) % len(GESTURES)]
                collecting = False
                print(f"  Selected: [{selected_gesture['id']}] {selected_gesture['name']}")

            elif chr(key).lower() == 'p':
                if selected_gesture is None:
                    selected_gesture = GESTURES[-1]
                else:
                    selected_gesture = GESTURES[(selected_gesture["id"] - 1) % len(GESTURES)]
                collecting = False
                print(f"  Selected: [{selected_gesture['id']}] {selected_gesture['name']}")

    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("\nCollection complete. Files saved to:", RAW_DATA_DIR)


if __name__ == "__main__":
    main()
