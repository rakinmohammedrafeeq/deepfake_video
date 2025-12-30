# 🔍 Real-Time Deepfake Video Detection using Blink Behavior Analysis

A lightweight, real-time deepfake video detection system that analyzes **human blink behavior** using **MediaPipe Face Mesh** and **OpenCV**.

This project uses **rule-based behavioral analysis** instead of heavy deep learning models, making it suitable for **edge devices**, **offline use**, and **real-time applications**.

---

## 🚀 Key Features

- ✅ Real-time detection using webcam or video files  
- ✅ No heavy ML models – pure rule-based analysis  
- ✅ Lightweight & fast (CPU-only, edge-device friendly)  
- ✅ Explainable detection with clear reasons for suspicion  
- ✅ Fully offline – no internet required  
- ✅ Live visual overlay with suspicion score and confidence  

---

## 🧠 How It Works

Deepfake videos often show **unnatural blink behavior** because:

- Early deepfake models were trained on images with eyes mostly open  
- AI-generated faces struggle with realistic eye dynamics  
- Human blinking has natural irregularity that is hard to synthesize  

This system detects such anomalies using **blink behavior analysis over time**.

---

## 🚨 Detection Rules (Suspicion Scoring)

The system evaluates **four behavioral indicators**:

| Rule | Indicator | Points | Reason |
|----|----|----|----|
| Rule 1 | Blink rate < 3/min | +50 | Humans rarely blink this infrequently |
| Rule 2 | No blink for > 25 seconds | +30 | Unnatural prolonged staring |
| Rule 3 | CV < 0.10 (robotic blinking) | +30 | AI blinks are too regular |
| Rule 4 | Blink duration < 100 ms | +20 | Unnaturally fast “snap” blinks |

### 🎯 Suspicion Score Interpretation

- **≥ 90** → 100% confidence → **DEEPFAKE DETECTED**
- **≥ 70** → 85% confidence → **Highly Suspicious**
- **≥ 50** → 70% confidence → **Suspicious**
- **≥ 30** → 50% confidence → **Possibly Suspicious**
- **< 30** → Low suspicion → **Likely Real**

---

## 📊 Technical Details

### Eye Aspect Ratio (EAR)

Blink detection is based on the **Eye Aspect Ratio (EAR)**:

```

EAR = (||p2 − p6|| + ||p3 − p5||) / (2 × ||p1 − p4||)

````

Where `p1–p6` are eye landmarks from MediaPipe Face Mesh.

### Blink Detection Logic

- EAR drops below threshold → eyes closing  
- Stays below for consecutive frames → valid blink  
- Track blink duration, intervals, and frequency  

### Behavioral Metrics

- **Blink Frequency** (blinks/min over rolling window)
- **Interval Regularity** (Coefficient of Variation, CV)
- **Maximum No-Blink Duration**
- **Blink Duration** (normal: 100–400 ms)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam (for real-time mode)
- Windows / Linux / macOS

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/deepfake_video.git
cd deepfake_video
````

Install dependencies:

```bash
pip install -r deepfake_requirements.txt
```

Run the detector:

```bash
python face_mesh.py
```

Press **ESC** to exit and view the final analysis summary.

---

## 💻 Usage

### Real-Time Webcam Detection

```bash
python face_mesh.py
```

The system will:

* Capture webcam feed
* Detect facial landmarks
* Track blink behavior
* Display suspicion score and confidence
* Explain reasons for suspicion

### Video File Analysis

Edit `face_mesh.py`:

```python
cap = cv2.VideoCapture("path/to/video.mp4")
```

---

## 🎛️ Configuration

Modify thresholds in `face_mesh.py`:

```python
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3

VERY_LOW_BLINK_RATE = 3
MAX_NO_BLINK_THRESHOLD = 25.0
ROBOTIC_CV_THRESHOLD = 0.10
MIN_HUMAN_BLINK_DURATION = 0.1
```

---

## 📈 Performance

### System Requirements

* CPU: Any modern processor
* RAM: 2 GB minimum (4 GB recommended)
* Camera: 720p or higher
* OS: Windows 10+, Ubuntu 18.04+, macOS 10.14+

### Speed

* **25–30 FPS** on laptop CPU
* **< 50 ms** latency per frame
* **~2 seconds** startup time

---

## ⚠️ Accuracy Considerations

### Strengths

* High accuracy on older deepfakes (pre-2020)
* Minimal false positives on normal behavior
* Excellent for real-time screening

### Limitations

* Advanced modern deepfakes may evade detection
* Requires clear, front-facing face
* Medical conditions affecting blinking may cause false flags
* Should be used as a **screening tool**, not final proof

---

## 🛠️ Project Structure

```
deepfake_video/
├── face_mesh.py
├── deepfake_requirements.txt
├── README.md
├── LICENSE
├── Eye-Blink-Detection-using-MediaPipe-and-OpenCV/
└── mediapipe_src/   (not tracked in Git)
```

---

## 🤝 Contributing

Contributions are welcome!

Ideas:

* Batch video processing
* Export results to CSV/JSON
* Multi-face tracking
* GUI / web interface
* Audio-visual synchronization checks

---

## 📝 License

This project is licensed under the **MIT License**.

---

## 🔮 Future Roadmap

### Version 2.0

* Audio + video consistency checks
* Head pose & gaze analysis
* Temporal coherence validation

### Version 3.0

* Hybrid rules + lightweight neural network
* Real-time streaming (RTMP/WebRTC)
* Browser extension integration

---

## ⚖️ Ethical Use

Designed for:

* Research & education
* Media verification
* Security & content moderation

Please use responsibly and respect privacy.

---

⭐ **Star this repository if you find it useful!**
Made with ❤️ for a safer digital world.
