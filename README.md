# Real-Time AI Sign Language to Text & Speech Translator

> **⚠️ Prototype Notice**: This application recognises a **controlled set of 10 predefined static hand gestures**. It is **not** a complete natural-language sign language interpreter. It is an accessibility-focused prototype for basic communication assistance.

---

## Overview

A fully offline, real-time desktop application that:
- Uses your **webcam** + **MediaPipe** to detect hand landmarks
- Passes the landmarks through a trained **Random Forest** classifier
- Converts confirmed gestures into **text** (sentence builder)
- Speaks confirmed words aloud via **pyttsx3** (offline TTS)
- Runs entirely **locally** — no internet or cloud APIs required

---

## Features

| Feature | Details |
|---|---|
| Real-time gesture detection | ~30 FPS, MediaPipe hand tracking |
| 10 predefined gestures | Hello, Thank You, Yes, No, Help, Stop, Please, Sorry, Good, I Love You |
| ML classifier | Random Forest, 63-feature landmark vectors |
| Confidence filtering | Only accepts predictions ≥ 80% confidence |
| Temporal smoothing | Majority vote over 10 frames prevents flickering |
| Sentence builder | Word cooldown prevents duplicate spam |
| Offline TTS | pyttsx3, background thread (UI never freezes) |
| Modern dark UI | CustomTkinter, colour-coded status indicators |
| Extensible | Add new gestures without changing core code |

---

## Real-World Applications

- Basic communication for individuals with speech impairments
- Accessibility interfaces for public kiosks
- Educational tool for learning sign language gestures
- Research baseline for gesture recognition systems

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Hand Detection | MediaPipe Hands |
| Computer Vision | OpenCV |
| ML Classifier | scikit-learn RandomForest |
| Feature Processing | NumPy |
| Model Storage | joblib / pickle |
| Text-to-Speech | pyttsx3 |
| Desktop UI | CustomTkinter |
| Testing | pytest |

---

## Project Architecture

```
Sign Lan/
├── app.py                          # Entry point
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                        # Per-gesture CSV files (collected data)
│   └── processed/
│       └── dataset.csv             # Combined training dataset
│
├── models/
│   └── gesture_model.pkl           # Trained Random Forest model
│
├── src/
│   ├── gesture_config.py           # ← ALL configuration constants
│   ├── camera.py                   # Thread-safe webcam abstraction
│   ├── hand_detector.py            # MediaPipe wrapper
│   ├── feature_extractor.py        # Landmark normalisation → 63-dim vector
│   ├── gesture_classifier.py       # Model load + predict with confidence
│   ├── prediction_stabilizer.py    # Temporal majority-vote smoother
│   ├── sentence_builder.py         # Word accumulation + cooldown
│   ├── speech_engine.py            # pyttsx3 background TTS thread
│   └── utils.py                    # Logging, image helpers, FPS counter
│
├── training/
│   ├── collect_data.py             # Interactive data collection
│   ├── train_model.py              # Training pipeline + metrics
│   └── evaluate_model.py          # Evaluation + confusion matrix
│
├── ui/
│   ├── main_window.py              # Root window + pipeline wiring
│   ├── camera_panel.py             # Live feed widget
│   └── control_panel.py           # Buttons + status + sentence display
│
└── tests/
    ├── test_feature_extractor.py
    ├── test_gesture_classifier.py
    ├── test_sentence_builder.py
    ├── test_prediction_stabilizer.py
    └── test_config.py
```

---

## Installation

### Requirements
- Python 3.11 or newer
- A working webcam

### Step 1: Clone / open the project folder

```bash
cd "Sign Lan"
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Collection

You must collect training data **before** training the model.

```bash
python training/collect_data.py
```

**Controls:**
| Key | Action |
|---|---|
| `0` – `9` | Select gesture by number |
| `SPACE` | Start / Pause collection |
| `Q` / `ESC` | Quit |

**Tips:**
- Collect ~300–400 samples per gesture
- Vary your hand distance and slight angle variations
- Ensure consistent lighting
- The right panel shows collection progress for all gestures
- A ✓ appears when a gesture reaches the target count

**Gesture key mapping:**
| Key | Gesture |
|---|---|
| 0 | Hello |
| 1 | Thank You |
| 2 | Yes |
| 3 | No |
| 4 | Help |
| 5 | Stop |
| 6 | Please |
| 7 | Sorry |
| 8 | Good |
| 9 | I Love You |

---

## Model Training

After collecting data for all gestures:

```bash
python training/train_model.py
```

This will:
1. Load all CSV files from `data/raw/`
2. Run 5-fold cross-validation
3. Train the final Random Forest model
4. Print accuracy, precision, recall, F1 score
5. Save the model to `models/gesture_model.pkl`

---

## Model Evaluation

```bash
python training/evaluate_model.py
```

Displays:
- Overall accuracy, macro precision, recall, F1
- Per-class classification report
- Confusion matrix (terminal + saved image at `models/confusion_matrix.png`)

---

## Running the Application

```bash
python app.py
```

**UI Controls:**
| Button | Action |
|---|---|
| ▶ Start Camera | Opens webcam and begins recognition |
| ■ Stop Camera | Stops webcam cleanly |
| 🔊 Speak Sentence | Reads the entire built sentence aloud |
| ⌫ Undo Word | Removes the last word from the sentence |
| 🗑 Clear All | Clears the entire sentence |
| ✕ Exit | Gracefully shuts down all resources |

---

## How Gesture Recognition Works

```
1. Webcam captures frame at ~30 FPS
2. MediaPipe detects 21 hand landmarks (x, y, z per point)
3. Feature Extractor normalises landmarks:
   a. Translate all points relative to wrist (landmark 0)
   b. Divide by max absolute value → all values in [-1, 1]
   c. Flatten → 63-dimensional feature vector
4. Random Forest classifier outputs class probabilities
5. Confidence Filter: reject if max probability < 80%
6. Prediction Stabilizer: require 8/10 frames to agree on same gesture
7. Sentence Builder: add confirmed word (with 2s cooldown)
8. TTS Engine: speak the word in background thread
```

---

## Adding New Gestures

1. Open `src/gesture_config.py`
2. Add a new entry to the `GESTURES` list:
   ```python
   {"id": 10, "name": "Please Wait", "spoken": "Please Wait"},
   ```
3. Re-run data collection:  `python training/collect_data.py`
4. Re-run training:         `python training/train_model.py`

No other code needs to change.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot open camera` | Check webcam is connected; change `CAMERA_INDEX` in `gesture_config.py` |
| `Model not found` | Run `collect_data.py` then `train_model.py` |
| Poor accuracy | Collect more varied samples; ensure consistent lighting |
| TTS not working | Check `pyttsx3` is installed; try `pip install pyttsx3` |
| UI doesn't open | Ensure `customtkinter` and `Pillow` are installed |
| `mediapipe` errors | Use Python 3.11; `pip install mediapipe==0.10.9` |
| Flickering predictions | Increase `STABILITY_MIN_AGREE` in `gesture_config.py` |
| Words added too fast | Increase `WORD_COOLDOWN_SECONDS` in `gesture_config.py` |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: feature extraction, classifier confidence filtering, sentence builder cooldown, prediction stabilizer, and config consistency. Tests do **not** require a webcam or a trained model.

---

## Limitations

- **Static gestures only**: Does not recognise motion-based or dynamic signs
- **Single hand**: Designed for one-hand detection (most reliable)
- **Lighting sensitive**: Works best in consistent, well-lit conditions
- **10 gestures**: Prototype vocabulary only — not a full sign language system
- **Training data quality**: Accuracy depends entirely on the quality of collected samples
- **Not a translation system**: Cannot interpret natural sign language sentences or grammar

---

## Future Improvements

- [ ] Add sequence/motion-based gesture recognition (LSTM / Transformer)
- [ ] Two-hand support
- [ ] Expand vocabulary with community-contributed data
- [ ] Real-time confidence visualisation graph
- [ ] Export sentence as audio file
- [ ] Multi-language TTS support
- [ ] Mobile app port (MediaPipe + TFLite)

---

## License

MIT License — Free for personal and educational use.
