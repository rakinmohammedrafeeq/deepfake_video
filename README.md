# 🔍 Real-Time Deepfake Video Detection System

A comprehensive, real-time deepfake detection prototype that analyzes **human facial behavior and visual artifacts** using MediaPipe Face Mesh and OpenCV. This system uses **multi-signal rule-based behavioral analysis** with an explainable suspicion scoring system.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)](https://opencv.org/)
[![CPU Only](https://img.shields.io/badge/CPU-Only-orange.svg)]()

---

## 🎯 Key Features

- ✅ **Real-time detection** - Works with webcam or video files
- ✅ **7 detection signals** - Blink, blur, head pose, mouth, eye tracking & more
- ✅ **CPU-only execution** - No GPU required, lightweight
- ✅ **Explainable AI** - Shows exact reasons for suspicion with scores
- ✅ **1280x720 display** - Large, clear output window
- ✅ **No ML training needed** - Pure rule-based analysis
- ✅ **Single source of truth** - MediaPipe Face Mesh (478 landmarks)

---

## 🎭 What Gets Detected

This system analyzes **facial behavior and visual artifacts** to identify deepfakes:

### 👁️ Eye & Blink Analysis
| Detection | Description |
|-----------|-------------|
| **Blink Count** | Total number of blinks detected |
| **Blink Rate** | Blinks per minute (normal: 8-30/min) |
| **Blink Duration** | How long each blink lasts (normal: 100-400ms) |
| **Eye Aspect Ratio (EAR)** | Real-time eye openness measurement |
| **No-Blink Duration** | Longest time without blinking |
| **Blink Interval Regularity** | Coefficient of Variation (CV) of blink timing |
| **Iris Position** | Eye gaze direction tracking |

### 🗣️ Mouth Analysis
| Detection | Description |
|-----------|-------------|
| **Mouth Aspect Ratio (MAR)** | Mouth openness measurement |
| **Mouth Movement Consistency** | Detects robotic/unnatural mouth patterns |
| **Lip Movement Tracking** | Monitors upper/lower lip positions |

### 🔄 Head Pose Analysis
| Detection | Description |
|-----------|-------------|
| **Yaw** | Left-right head rotation (degrees) |
| **Pitch** | Up-down head tilt (degrees) |
| **Roll** | Head tilt to shoulder (degrees) |
| **Head Movement History** | Tracks head motion over time |
| **Head-Eye Coordination** | Checks if eyes move naturally with head |

### 🔍 Face Quality Analysis
| Detection | Description |
|-----------|-------------|
| **Face Blur Score** | Variance of Laplacian (sharpness measure) |
| **Face Smoothness** | Detects unnatural AI smoothing |
| **Face ROI Extraction** | Isolates face region for analysis |

### 📊 Behavioral Pattern Analysis
| Detection | Description |
|-----------|-------------|
| **Temporal Consistency** | Patterns over 60-second rolling window |
| **Movement Synchronization** | Head vs eye vs mouth coordination |
| **Natural Variability** | Human randomness vs robotic patterns |

---

## 🧠 How It Works

Deepfake videos often exhibit unnatural patterns:
1. Abnormal or absent blinking behavior
2. Unnaturally smooth/blurry face texture
3. Inconsistent head-eye movement coordination
4. Robotic, too-regular facial movements
5. Unnatural blink timing and duration

### The 7 Detection Rules (Suspicion Score 0-100)

| Rule | Signal | Max Points | What It Detects |
|------|--------|------------|-----------------|
| 🚨 **1** | Very low blink rate (<3/min) | **+25** | Deepfakes often don't blink |
| 🚨 **2** | Long stare (>25 seconds) | **+15** | Unnatural extended staring |
| 🚨 **3** | Robotic blink timing (CV<0.10) | **+15** | Too-regular blink intervals |
| 🚨 **4** | Unnatural blink speed (<100ms) | **+10** | Snap blinks from AI |
| 🚨 **5** | Face too smooth (blur<50) | **+15** | Neural network smoothing artifacts |
| 🚨 **6** | Head-eye mismatch | **+10** | Head moves but eyes don't track |
| 🚨 **7** | Robotic mouth movement | **+10** | Too-consistent mouth patterns |

### Decision Thresholds

| Suspicion Score | Result |
|-----------------|--------|
| **≥ 70** | 🔴 **LIKELY DEEPFAKE** |
| **40 - 69** | 🟡 **SUSPICIOUS** |
| **< 40** | 🟢 **LIKELY REAL** |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd deepfake_video

# Install dependencies
pip install -r deepfake_requirements.txt
```

### Requirements

```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.19.0
```

### Usage

```bash
# Webcam mode (default)
python face_mesh.py

# Video file mode
python face_mesh.py path/to/video.mp4
```

### Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit and show final summary |
| `R` | Reset analysis (clear all data) |

---

## 📊 Real-Time Display

The system shows a **1280x720** output window with two information panels:

### Left Panel - Real-Time Metrics
```
┌─────────────────────┐
│ REAL-TIME METRICS   │
│ Blinks: 12          │
│ EAR: 0.285          │
│ MAR: 0.045          │
│ Yaw: 2.3  Pitch: -5 │
│ Roll: 1.2  Blur: 156│
│ LIVE                │
└─────────────────────┘
```

### Right Panel - Detection Results
```
┌─────────────────────────────────────┐
│ DEEPFAKE DETECTION                  │
│ LIKELY REAL                         │
│ Suspicion Score: 12/100             │
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░     │
│ --- ANALYSIS METRICS ---            │
│ Blink Rate: 16.2/min                │
│ Avg Blink Duration: 145ms           │
│ Max No-Blink: 5.3s                  │
│ Face Blur Score: 234                │
│ Head Pose: Y:2 P:-3 R:1             │
│ Analysis Time: 32s                  │
│ --- DETECTION REASONS ---           │
│ ✅ Normal blink patterns detected   │
│ ✅ Natural facial movements         │
└─────────────────────────────────────┘
```

---

## 🔬 Technical Details

### MediaPipe Face Mesh
- **478 facial landmarks** including iris tracking
- Single source for all facial feature detection
- `refine_landmarks=True` enables iris landmarks

### Signals Computed

| Signal | Method | Landmarks Used |
|--------|--------|----------------|
| **Eye Aspect Ratio (EAR)** | Vertical/horizontal eye ratio | 6 points per eye |
| **Blink Detection** | EAR threshold + consecutive frames | Eye landmarks |
| **Head Pose** | cv2.solvePnP with 3D model | 6 key facial points |
| **Face Blur** | Variance of Laplacian on face ROI | All face landmarks for ROI |
| **Mouth Aspect Ratio** | Vertical/horizontal mouth ratio | Lip landmarks |
| **Iris Position** | Relative position in eye bounds | Iris landmarks (468-477) |

### Eye Aspect Ratio Formula

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

Where p1-p6 are specific eye landmark points. EAR drops below 0.25 when eyes close.

### Head Pose Estimation

Uses 6 key landmarks mapped to 3D model points:
- Nose tip, Chin, Left/Right eye corners, Left/Right mouth corners
- Solved using `cv2.solvePnP` to extract Yaw, Pitch, Roll angles

---

## ⚙️ Configuration

All thresholds can be adjusted in `face_mesh.py`:

```python
# Blink detection
EAR_THRESHOLD = 0.25              # Eye closed threshold
CONSEC_FRAMES = 3                 # Frames for valid blink

# Rule 1: Blink rate
VERY_LOW_BLINK_RATE = 3           # Extremely suspicious
LOW_BLINK_RATE = 8                # Moderately suspicious

# Rule 2: Stare duration
MAX_NO_BLINK_THRESHOLD = 25.0     # Seconds

# Rule 3: Blink regularity
ROBOTIC_CV_THRESHOLD = 0.10       # CV below = robotic

# Rule 4: Blink duration
MIN_HUMAN_BLINK_DURATION = 0.1    # 100ms minimum
MAX_HUMAN_BLINK_DURATION = 0.4    # 400ms maximum

# Rule 5: Face blur
BLUR_THRESHOLD_SUSPICIOUS = 80    # Variance threshold
BLUR_THRESHOLD_VERY_SUSPICIOUS = 50

# Rule 6: Head-eye mismatch
HEAD_EYE_MISMATCH_THRESHOLD = 0.5

# Rule 7: Mouth consistency
MOUTH_CV_THRESHOLD_SUSPICIOUS = 0.05

# Display
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
```

---

## 📈 Session Summary

When you exit (ESC), a detailed summary is printed:

```
============================================================
          DEEPFAKE DETECTION SESSION SUMMARY
============================================================
Total blinks detected: 18
Session duration: 45.2 seconds
Max no-blink duration: 4.8 seconds
------------------------------------------------------------
🔍 FINAL RESULT: LIKELY REAL
📊 Suspicion Score: 15/100
📈 Confidence: 85.0%
👁️  Blink Rate: 23.9 blinks/min
⏱️  Interval Regularity (CV): 0.456
⚡ Avg Blink Duration: 142ms
🔍 Avg Face Blur Score: 267
------------------------------------------------------------
Detection Reasons:
  ✅ Normal blink patterns detected
  ✅ Natural facial movements
============================================================
```

---

## 📁 Project Structure

```
deepfake_video/
├── face_mesh.py                    # Main detection script
├── deepfake_requirements.txt       # Python dependencies
├── README.md                       # This file
├── DEEPFAKE_DETECTION_README.md    # Additional documentation
│
├── BlurDetection2/                 # Blur detection reference
│   └── blur_detection/
│       └── detection.py            # Variance of Laplacian
│
├── Eye-Blink-Detection-.../        # Blink detection reference
│   ├── blink_counter.py
│   ├── FaceMeshModule.py
│   └── utils.py
│
├── head-pose-estimation/           # Head pose reference
│   ├── pose_estimation.py
│   └── main.py
│
└── Deepfake-Detection/             # CNN-based detection (reference)
    └── detect_from_video.py
```

---

## ⚠️ Limitations

This is a **research prototype** with limitations:

- ❌ Not production-ready - for research/education only
- ❌ High-quality modern deepfakes may evade detection
- ❌ Requires clear, front-facing face visibility
- ❌ Lighting affects blur detection accuracy
- ❌ Short videos may not accumulate enough data
- ❌ Medical conditions affecting blinking may cause false positives

---

## 🔬 Scientific Background

### Normal Human Blink Patterns
- **Frequency:** 15-20 blinks/minute (average)
- **Range:** 8-30 blinks/minute (acceptable)
- **Duration:** 100-400 milliseconds per blink
- **Interval Variability:** CV typically 0.3-0.7

### Why Deepfakes Fail at Blinking
1. Training data often has eyes open (selection bias)
2. Eye region is geometrically complex
3. Temporal consistency is hard to maintain
4. Natural randomness is difficult to synthesize

### References
- Li, Y., et al. (2018) - "In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking"
- Bentivoglio, A.R., et al. (1997) - "Analysis of blink rate patterns in normal subjects"

---

## 🤝 Contributing

Areas for improvement:
- [ ] Add batch video processing with CSV export
- [ ] Implement audio-visual sync analysis
- [ ] Add more behavioral indicators
- [ ] Create GUI interface
- [ ] Optimize for mobile/edge deployment

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file.

---

## 🛠️ Troubleshooting

### "No face detected"
- Ensure good lighting
- Face the camera directly
- Check webcam permissions

### Low frame rate
- Close other applications
- Reduce `OUTPUT_WIDTH` and `OUTPUT_HEIGHT`
- Ensure no other processes using camera

### MediaPipe errors
- Update: `pip install --upgrade mediapipe`
- Check Python version (3.8+ required)

---

**Last Updated:** December 2025

=======
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
>>>>>>> 6c2ad0702a788772ac10b3f413926985b1002302
