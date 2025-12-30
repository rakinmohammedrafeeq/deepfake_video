import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

print("MediaPipe loaded from:", mp.__file__)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ==================== BLINK DETECTION CONFIGURATION ====================
# Eye landmark indices for EAR (Eye Aspect Ratio) calculation
RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]

# Full eye contour landmarks for visualization
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Blink detection thresholds
EAR_THRESHOLD = 0.25  # Eye aspect ratio below this indicates closed eyes
CONSEC_FRAMES = 3     # Number of consecutive frames eyes must be closed to count as blink

# Visualization colors (BGR format)
GREEN_COLOR = (86, 241, 13)   # Eyes open / Likely Real
RED_COLOR = (30, 46, 209)     # Eyes closed / Suspicious
YELLOW_COLOR = (0, 255, 255)  # Warning / Analyzing
WHITE_COLOR = (255, 255, 255)

# Blink detection state variables
blink_counter = 0
frame_counter = 0

# ==================== DEEPFAKE DETECTION CONFIGURATION ====================
# Rolling time window for blink analysis (in seconds)
ANALYSIS_WINDOW = 60.0  # 60 seconds rolling window

# ==========================================================================
# SUSPICION SCORE THRESHOLDS (Rule-Based Deepfake Detection)
# ==========================================================================

# 🚨 Rule 1: Very low blink rate (major signal) - adds 50 points
# Humans almost never blink this little - strong deepfake indicator
VERY_LOW_BLINK_RATE = 3     # Blinks per minute - extremely suspicious
LOW_BLINK_RATE = 8          # Blinks per minute - moderately suspicious

# 🚨 Rule 2: Long no-blink duration - adds 30 points
# Humans cannot stare that long naturally
MAX_NO_BLINK_THRESHOLD = 25.0  # seconds without blinking is suspicious

# 🚨 Rule 3: Perfect blink timing (robotic) - adds 30 points
# AI-generated blinking is often evenly spaced
ROBOTIC_CV_THRESHOLD = 0.10    # Coefficient of Variation below this = robotic

# 🚨 Rule 4: Very short blinks - adds 20 points
# Deepfakes sometimes "snap" eyes shut unnaturally
MIN_HUMAN_BLINK_DURATION = 0.1  # seconds - human blinks are 100-400ms
MAX_HUMAN_BLINK_DURATION = 0.4  # seconds

# Normal human ranges for reference
NORMAL_MIN_BLINK_RATE = 8   # Blinks per minute
NORMAL_MAX_BLINK_RATE = 30  # Blinks per minute

# Minimum data required before making a detection decision
MIN_BLINKS_FOR_ANALYSIS = 3
MIN_ANALYSIS_TIME = 10.0  # seconds

# Deepfake detection state variables
blink_timestamps = deque(maxlen=100)  # Store timestamps of recent blinks
blink_durations = deque(maxlen=100)   # Store duration of each blink
last_blink_time = None                # Track time since last blink
max_no_blink_duration = 0.0           # Maximum time without blinking
blink_start_time = None               # Track when current blink started
analysis_start_time = None
detection_result = "Analyzing..."
detection_confidence = 0.0
suspicion_reasons = []


def calculate_ear(eye_landmarks, landmarks, frame_shape):
    """
    Calculate the Eye Aspect Ratio (EAR) for given eye landmarks.

    The EAR is calculated using the formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1-p6 are specific points around the eye:
    - p1, p4: horizontal eye corners (outer and inner)
    - p2, p6: upper and lower eyelid points (outer pair)
    - p3, p5: upper and lower eyelid points (inner pair)

    Args:
        eye_landmarks: List of 6 landmark indices for EAR calculation
        landmarks: MediaPipe face landmarks object
        frame_shape: Tuple of (height, width) for coordinate conversion

    Returns:
        float: The calculated eye aspect ratio
    """
    h, w = frame_shape[:2]

    # Extract (x, y) pixel coordinates for each landmark
    def get_point(idx):
        lm = landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])

    # Get the 6 eye points
    p1 = get_point(eye_landmarks[0])  # Outer corner
    p2 = get_point(eye_landmarks[1])  # Upper outer
    p3 = get_point(eye_landmarks[2])  # Upper inner
    p4 = get_point(eye_landmarks[3])  # Inner corner
    p5 = get_point(eye_landmarks[4])  # Lower inner
    p6 = get_point(eye_landmarks[5])  # Lower outer

    # Calculate euclidean distances
    A = np.linalg.norm(p2 - p6)  # Vertical distance (outer)
    B = np.linalg.norm(p3 - p5)  # Vertical distance (inner)
    C = np.linalg.norm(p1 - p4)  # Horizontal distance

    # Calculate EAR
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear


def update_blink_count(ear, current_time):
    """
    Update blink counter based on current eye aspect ratio.
    Also tracks blink duration and time between blinks.

    Logic:
    - If EAR is below threshold, eyes are closing/closed
    - If EAR returns above threshold and enough consecutive frames were counted,
      increment blink counter and record blink duration

    Args:
        ear: Current eye aspect ratio
        current_time: Current timestamp for duration tracking

    Returns:
        bool: True if a new blink was detected
    """
    global blink_counter, frame_counter, blink_start_time, last_blink_time, max_no_blink_duration

    blink_detected = False

    if ear < EAR_THRESHOLD:
        # Eyes are closing/closed
        if frame_counter == 0:
            # Blink just started
            blink_start_time = current_time
        frame_counter += 1
    else:
        # Eyes are open
        if frame_counter >= CONSEC_FRAMES:
            # Valid blink completed
            blink_counter += 1
            blink_detected = True

            # Calculate and store blink duration
            if blink_start_time is not None:
                blink_duration = current_time - blink_start_time
                blink_durations.append(blink_duration)

            # Track max time without blinking
            if last_blink_time is not None:
                no_blink_duration = current_time - last_blink_time
                if no_blink_duration > max_no_blink_duration:
                    max_no_blink_duration = no_blink_duration

            last_blink_time = current_time

        frame_counter = 0
        blink_start_time = None

    return blink_detected


def draw_eye_landmarks(frame, landmarks, eye_indices, color, frame_shape):
    """
    Draw landmarks around the eyes on the frame.

    Args:
        frame: Video frame to draw on
        landmarks: MediaPipe face landmarks object
        eye_indices: List of landmark indices for the eye
        color: BGR color tuple for drawing
        frame_shape: Tuple of (height, width) for coordinate conversion
    """
    h, w = frame_shape[:2]
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 2, color, cv2.FILLED)


# ==================== DEEPFAKE DETECTION FUNCTIONS ====================

def get_blinks_in_window(current_time):
    """
    Get blink timestamps within the rolling analysis window.

    Args:
        current_time: Current timestamp

    Returns:
        list: Timestamps of blinks within the window
    """
    window_start = current_time - ANALYSIS_WINDOW
    return [t for t in blink_timestamps if t >= window_start]


def calculate_blink_rate(blinks_in_window, window_duration):
    """
    Calculate blinks per minute from blinks in the window.

    Args:
        blinks_in_window: List of blink timestamps in window
        window_duration: Duration of the analysis window in seconds

    Returns:
        float: Blinks per minute
    """
    if window_duration <= 0:
        return 0.0
    blinks_per_second = len(blinks_in_window) / window_duration
    return blinks_per_second * 60.0


def calculate_interval_regularity(blinks_in_window):
    """
    Calculate the Coefficient of Variation (CV) of blink intervals.
    CV = standard_deviation / mean

    Low CV indicates unnaturally regular blink intervals (robotic).
    High CV indicates erratic blinking.
    Normal humans have moderate CV (natural variation).

    Args:
        blinks_in_window: List of sorted blink timestamps

    Returns:
        tuple: (coefficient_of_variation, mean_interval, std_interval)
               Returns (None, None, None) if insufficient data
    """
    if len(blinks_in_window) < 3:
        return None, None, None

    # Calculate intervals between consecutive blinks
    sorted_blinks = sorted(blinks_in_window)
    intervals = [sorted_blinks[i+1] - sorted_blinks[i]
                 for i in range(len(sorted_blinks) - 1)]

    if len(intervals) < 2:
        return None, None, None

    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)

    if mean_interval == 0:
        return None, None, None

    cv = std_interval / mean_interval
    return cv, mean_interval, std_interval


def analyze_blink_behavior(current_time):
    """
    Analyze blink behavior and determine if video is likely real or deepfake.

    Uses SUSPICION SCORE rule-based decision logic:
    🚨 Rule 1: Very low blink rate (<3/min) -> +50 points
    🚨 Rule 2: Long no-blink duration (>25s) -> +30 points
    🚨 Rule 3: Perfect blink timing (robotic CV) -> +30 points
    🚨 Rule 4: Very short blinks (<100ms) -> +20 points

    Args:
        current_time: Current timestamp

    Returns:
        tuple: (result_string, confidence, list_of_reasons, metrics_dict)
    """
    global analysis_start_time, max_no_blink_duration, last_blink_time

    if analysis_start_time is None:
        analysis_start_time = current_time

    elapsed_time = current_time - analysis_start_time
    blinks_in_window = get_blinks_in_window(current_time)

    # Use actual elapsed time or window duration, whichever is smaller
    effective_window = min(elapsed_time, ANALYSIS_WINDOW)

    # Calculate metrics
    blink_rate = calculate_blink_rate(blinks_in_window, effective_window)
    cv, mean_interval, std_interval = calculate_interval_regularity(blinks_in_window)

    # Calculate average blink duration
    avg_blink_duration = np.mean(list(blink_durations)) if len(blink_durations) > 0 else None

    # Calculate current no-blink duration (time since last blink)
    current_no_blink = 0.0
    if last_blink_time is not None:
        current_no_blink = current_time - last_blink_time
    elif analysis_start_time is not None:
        current_no_blink = elapsed_time  # No blinks yet, count from start

    metrics = {
        'blink_rate': blink_rate,
        'blinks_in_window': len(blinks_in_window),
        'elapsed_time': elapsed_time,
        'cv': cv,
        'mean_interval': mean_interval,
        'max_no_blink': max(max_no_blink_duration, current_no_blink),
        'avg_blink_duration': avg_blink_duration
    }

    # Not enough data yet
    if elapsed_time < MIN_ANALYSIS_TIME:
        return "Analyzing...", 0.0, ["Collecting data..."], metrics

    # ==========================================================================
    # SUSPICION SCORE CALCULATION (0-100 scale)
    # ==========================================================================
    suspicion_score = 0
    reasons = []

    # 🚨 Rule 1: Very low blink rate (major signal) - adds 50 points
    # Humans almost never blink this little - strong deepfake indicator
    if blink_rate < VERY_LOW_BLINK_RATE:
        suspicion_score += 50
        reasons.append(f"🚨 Very low blink rate: {blink_rate:.1f}/min (<{VERY_LOW_BLINK_RATE})")
    elif blink_rate < LOW_BLINK_RATE:
        suspicion_score += 25
        reasons.append(f"⚠️ Low blink rate: {blink_rate:.1f}/min (<{LOW_BLINK_RATE})")

    # 🚨 Rule 2: Long no-blink duration - adds 30 points
    # Humans cannot stare that long naturally
    effective_max_no_blink = max(max_no_blink_duration, current_no_blink)
    if effective_max_no_blink > MAX_NO_BLINK_THRESHOLD:
        suspicion_score += 30
        reasons.append(f"🚨 Long stare: {effective_max_no_blink:.1f}s without blinking")

    # 🚨 Rule 3: Perfect blink timing (robotic) - adds 30 points
    # AI-generated blinking is often evenly spaced
    if cv is not None and cv < ROBOTIC_CV_THRESHOLD:
        suspicion_score += 30
        reasons.append(f"🚨 Robotic blink pattern (CV={cv:.3f}, too regular)")
    elif cv is not None and cv < 0.15:
        suspicion_score += 15
        reasons.append(f"⚠️ Suspiciously regular blinks (CV={cv:.3f})")

    # 🚨 Rule 4: Very short blinks - adds 20 points
    # Deepfakes sometimes "snap" eyes shut unnaturally
    if avg_blink_duration is not None:
        if avg_blink_duration < MIN_HUMAN_BLINK_DURATION:
            suspicion_score += 20
            reasons.append(f"🚨 Unnatural blink speed: {avg_blink_duration*1000:.0f}ms (<100ms)")
        elif avg_blink_duration > MAX_HUMAN_BLINK_DURATION:
            suspicion_score += 10
            reasons.append(f"⚠️ Slow blinks: {avg_blink_duration*1000:.0f}ms (>400ms)")

    # ==========================================================================
    # FINAL CONFIDENCE MAPPING
    # ==========================================================================
    if suspicion_score >= 90:
        confidence = 1.0  # 100%
        result = "DEEPFAKE DETECTED"
    elif suspicion_score >= 70:
        confidence = 0.85  # 85%
        result = "Highly Suspicious (Deepfake)"
    elif suspicion_score >= 50:
        confidence = 0.70  # 70%
        result = "Suspicious (Deepfake)"
    elif suspicion_score >= 30:
        confidence = 0.50  # 50%
        result = "Possibly Suspicious"
    else:
        # Low suspicion = likely real
        confidence = max(0.30, (100 - suspicion_score) / 100)
        result = "Likely Real"
        if len(reasons) == 0:
            reasons.append("✅ Normal blink patterns detected")

    # Add suspicion score to metrics
    metrics['suspicion_score'] = suspicion_score

    return result, confidence, reasons, metrics


def draw_detection_overlay(frame, result, confidence, reasons, metrics):
    """
    Draw the deepfake detection results on the frame.

    Args:
        frame: Video frame to draw on
        result: Detection result string
        confidence: Confidence score (0-1)
        reasons: List of reason strings
        metrics: Dictionary of analysis metrics
    """
    h, w = frame.shape[:2]
    suspicion_score = metrics.get('suspicion_score', 0)

    # Choose colors based on result
    if "DEEPFAKE DETECTED" in result:
        result_color = RED_COLOR
        bg_color = (0, 0, 120)
    elif "Highly Suspicious" in result or "Suspicious" in result:
        result_color = RED_COLOR
        bg_color = (0, 0, 100)
    elif "Possibly" in result:
        result_color = YELLOW_COLOR
        bg_color = (0, 100, 100)
    elif "Likely Real" in result:
        result_color = GREEN_COLOR
        bg_color = (0, 100, 0)
    else:
        result_color = YELLOW_COLOR
        bg_color = (100, 100, 0)

    # Draw semi-transparent background panel
    overlay = frame.copy()
    panel_height = 180 + len(reasons) * 22
    cv2.rectangle(overlay, (w - 350, 10), (w - 10, panel_height), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw title
    cv2.putText(frame, "DEEPFAKE DETECTION", (w - 340, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE_COLOR, 2)

    # Draw detection result
    cv2.putText(frame, result, (w - 340, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)

    # Draw suspicion score
    cv2.putText(frame, f"Suspicion Score: {suspicion_score}/100", (w - 340, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE_COLOR, 1)

    # Draw suspicion score bar
    bar_width = int(280 * (suspicion_score / 100))
    bar_color = GREEN_COLOR if suspicion_score < 30 else (YELLOW_COLOR if suspicion_score < 50 else RED_COLOR)
    cv2.rectangle(frame, (w - 340, 95), (w - 340 + bar_width, 108), bar_color, cv2.FILLED)
    cv2.rectangle(frame, (w - 340, 95), (w - 60, 108), WHITE_COLOR, 1)

    # Draw confidence
    cv2.putText(frame, f"Confidence: {confidence*100:.0f}%", (w - 340, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE_COLOR, 1)

    # Draw metrics
    blink_rate = metrics.get('blink_rate', 0)
    elapsed = metrics.get('elapsed_time', 0)
    max_no_blink = metrics.get('max_no_blink', 0)
    cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min | Time: {elapsed:.0f}s", (w - 340, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE_COLOR, 1)
    cv2.putText(frame, f"Max No-Blink: {max_no_blink:.1f}s", (w - 340, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE_COLOR, 1)

    # Draw reasons
    y_offset = 185
    for reason in reasons[:4]:  # Show max 4 reasons
        # Truncate long reasons
        display_reason = reason[:45] + "..." if len(reason) > 45 else reason
        cv2.putText(frame, display_reason, (w - 340, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE_COLOR, 1)
        y_offset += 22


# ==================== MAIN VIDEO PROCESSING LOOP ====================
cap = cv2.VideoCapture(0)

# Initialize timing
analysis_start_time = time.time()

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh tesselation (lighter for clarity)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(200, 200, 200), thickness=1, circle_radius=1
                    )
                )

                # ========== BLINK DETECTION ==========
                # Calculate Eye Aspect Ratio for both eyes
                right_ear = calculate_ear(RIGHT_EYE_EAR, face_landmarks, frame.shape)
                left_ear = calculate_ear(LEFT_EYE_EAR, face_landmarks, frame.shape)
                avg_ear = (right_ear + left_ear) / 2.0

                # Update blink detection (with timing for duration tracking)
                blink_detected = update_blink_count(avg_ear, current_time)

                # Record blink timestamp for deepfake analysis
                if blink_detected:
                    blink_timestamps.append(current_time)

                # Set visualization color based on eye state
                eye_color = RED_COLOR if avg_ear < EAR_THRESHOLD else GREEN_COLOR

                # Draw eye landmarks with color indicating open/closed state
                draw_eye_landmarks(frame, face_landmarks, RIGHT_EYE, eye_color, frame.shape)
                draw_eye_landmarks(frame, face_landmarks, LEFT_EYE, eye_color, frame.shape)

                # ========== DEEPFAKE DETECTION ANALYSIS ==========
                result, confidence, reasons, metrics = analyze_blink_behavior(current_time)

                # Draw deepfake detection overlay (right side of screen)
                draw_detection_overlay(frame, result, confidence, reasons, metrics)

                # Display blink info (left side of screen)
                cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)

                # Visual feedback when blink is detected
                if blink_detected:
                    cv2.putText(frame, "BLINK!", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        else:
            # No face detected - show warning
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)

        cv2.imshow("Deepfake Video Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Print final analysis summary
print("\n" + "="*60)
print("          DEEPFAKE DETECTION SESSION SUMMARY")
print("="*60)
print(f"Total blinks detected: {blink_counter}")
print(f"Session duration: {time.time() - analysis_start_time:.1f} seconds")
print(f"Max no-blink duration: {max_no_blink_duration:.1f} seconds")

final_result, final_conf, final_reasons, final_metrics = analyze_blink_behavior(time.time())
print("-"*60)
print(f"🔍 FINAL RESULT: {final_result}")
print(f"📊 Suspicion Score: {final_metrics.get('suspicion_score', 0)}/100")
print(f"📈 Confidence: {final_conf*100:.1f}%")
print(f"👁️  Blink Rate: {final_metrics['blink_rate']:.1f} blinks/min")
if final_metrics['cv'] is not None:
    print(f"⏱️  Interval Regularity (CV): {final_metrics['cv']:.3f}")
if final_metrics.get('avg_blink_duration') is not None:
    print(f"⚡ Avg Blink Duration: {final_metrics['avg_blink_duration']*1000:.0f}ms")
print("-"*60)
print("Detection Reasons:")
for reason in final_reasons:
    print(f"  {reason}")
print("="*60)
