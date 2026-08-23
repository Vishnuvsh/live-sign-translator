"""
app.py
======
Entry point for the Real-Time AI Sign Language Translator.

Run with:
    python app.py

Prerequisites:
    1. pip install -r requirements.txt
    2. python training/collect_data.py    (collect gesture data)
    3. python training/train_model.py     (train the classifier)
    4. python app.py                      (launch the application)
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


def check_dependencies() -> None:
    """Verify required packages are installed before launching."""
    missing = []
    packages = {
        "cv2":           "opencv-python",
        "mediapipe":     "mediapipe",
        "numpy":         "numpy",
        "sklearn":       "scikit-learn",
        "pyttsx3":       "pyttsx3",
        "customtkinter": "customtkinter",
        "PIL":           "Pillow",
        "joblib":        "joblib",
    }
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("\n[ERROR] Missing dependencies:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print(
            "\nOr install everything at once:\n"
            "  pip install -r requirements.txt\n"
        )
        sys.exit(1)


def main() -> None:
    check_dependencies()

    # Add project root to path so all src/ imports resolve correctly
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from ui.main_window import MainWindow
        app = MainWindow()
        app.run()
    except Exception as exc:
        print(f"\n[FATAL ERROR] The application encountered an unexpected error:\n")
        traceback.print_exc()
        print(
            "\nTroubleshooting:\n"
            "  1. Ensure webcam is connected and not in use.\n"
            "  2. Ensure model exists: python training/train_model.py\n"
            "  3. Check requirements.txt is installed: pip install -r requirements.txt\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
