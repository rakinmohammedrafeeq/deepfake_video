# 🚀 Quick Start Guide - Deepfake Video Detection

## 5-Minute Setup

### Step 1: Prerequisites Check
```bash
# Check Python version (need 3.8+)
python --version

# Should output: Python 3.8.x or higher
```

### Step 2: Install Dependencies
```bash
# Install required packages
pip install opencv-python mediapipe numpy

# Or use the requirements file
pip install -r deepfake_requirements.txt
```

### Step 3: Run the Detector
```bash
# Start real-time detection
python face_mesh.py
```

### Step 4: Use the System

**During Detection:**
- Position your face in front of the webcam
- Ensure good lighting and front-facing view
- Let it analyze for at least 10 seconds
- Watch the suspicion score and metrics in real-time

**Reading the Display:**

**Left Side (Blink Info):**
- `Blinks: X` - Total blinks detected
- `EAR: 0.XX` - Current Eye Aspect Ratio
- `BLINK!` - Appears when blink detected

**Right Side (Detection Panel):**
- `Suspicion Score: X/100` - Higher = more suspicious
- `Confidence: X%` - Detection confidence
- `Blink Rate: X/min` - Current blink frequency
- `Max No-Blink: Xs` - Longest stare duration
- Reasons list - Why flagged as suspicious

**Exit:**
- Press `ESC` key to stop
- View detailed summary in terminal

---

## Understanding Results

### 🟢 Likely Real (Suspicion Score < 30)
```
Normal blink patterns detected
✅ Safe to proceed
```

### 🟡 Possibly Suspicious (30-49 points)
```
Some unusual patterns
⚠️ Review manually
```

### 🟠 Suspicious (50-69 points)
```
Multiple red flags detected
⚠️ Likely synthetic content
```

### 🔴 Highly Suspicious (70-89 points)
```
Strong indicators of deepfake
🚨 High confidence fake
```

### ⛔ DEEPFAKE DETECTED (90+ points)
```
Overwhelming evidence of synthesis
🚨 Almost certain deepfake
```

---

## Testing with Video Files

**Method 1: Modify the script**
```python
# Open face_mesh.py and find line ~471
# Change from:
cap = cv2.VideoCapture(0)

# To:
cap = cv2.VideoCapture("path/to/video.mp4")
```

**Method 2: Command line argument (if you add argparse)**
```bash
python face_mesh.py --video "path/to/video.mp4"
```

---

## Troubleshooting

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "No module named 'mediapipe'"
```bash
pip install mediapipe
```

### Webcam not opening
```python
# Try different camera indices in face_mesh.py
cap = cv2.VideoCapture(0)  # Default
cap = cv2.VideoCapture(1)  # External webcam
cap = cv2.VideoCapture(2)  # Another camera
```

### Low FPS / Slow performance
- Close other applications using the camera
- Ensure good lighting (helps face detection)
- Update your graphics drivers
- Consider reducing video resolution

### No face detected
- Ensure face is clearly visible
- Check lighting conditions
- Face should be front-facing
- Remove obstructions (masks, glasses may affect accuracy)

---

## Adjusting Sensitivity

**More Lenient (Fewer False Positives):**
```python
# In face_mesh.py, modify these values:
VERY_LOW_BLINK_RATE = 2           # From 3
MAX_NO_BLINK_THRESHOLD = 30.0     # From 25.0
ROBOTIC_CV_THRESHOLD = 0.08       # From 0.10
MIN_ANALYSIS_TIME = 15.0          # From 10.0
```

**More Strict (Catch More Suspicious Videos):**
```python
VERY_LOW_BLINK_RATE = 5           # From 3
MAX_NO_BLINK_THRESHOLD = 20.0     # From 25.0
ROBOTIC_CV_THRESHOLD = 0.15       # From 0.10
MIN_BLINKS_FOR_ANALYSIS = 5       # From 3
```

---

## Example Output

**Terminal Summary:**
```
============================================================
          DEEPFAKE DETECTION SESSION SUMMARY
============================================================
Total blinks detected: 23
Session duration: 67.3 seconds
Max no-blink duration: 8.4 seconds
------------------------------------------------------------
🔍 FINAL RESULT: Likely Real
📊 Suspicion Score: 15/100
📈 Confidence: 85%
👁️  Blink Rate: 20.5 blinks/min
⏱️  Interval Regularity (CV): 0.447
⚡ Avg Blink Duration: 235ms
------------------------------------------------------------
Detection Reasons:
  ✅ Normal blink patterns detected
============================================================
```

---

## Next Steps

1. **Test with known real videos** - Verify low suspicion scores
2. **Test with deepfake samples** - Check high suspicion scores
3. **Adjust thresholds** - Tune for your specific use case
4. **Integrate into workflow** - Use as pre-screening tool
5. **Read full README** - Understand technical details

---

## Key Metrics to Watch

| Metric | Normal Range | Suspicious Range |
|--------|--------------|------------------|
| Blink Rate | 8-30 /min | <3 or >40 /min |
| Max No-Blink | <20 seconds | >25 seconds |
| CV (Regularity) | 0.3-0.7 | <0.10 or >1.5 |
| Blink Duration | 100-400ms | <100ms or >500ms |

---

## Getting Help

- **Documentation:** See `DEEPFAKE_README.md` for full details
- **Issues:** Report bugs on GitHub Issues
- **Configuration:** Check the configuration section in README
- **Community:** Join discussions for tips and support

---

**Happy Detecting! 🔍**

Remember: This is a screening tool. Always verify suspicious results through multiple methods.

