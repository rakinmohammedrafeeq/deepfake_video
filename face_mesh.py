import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math

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

# ==================== HEAD POSE ESTIMATION LANDMARKS ====================
# Key facial landmarks for head pose estimation using solvePnP
# These correspond to: nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# 3D model points for head pose estimation (approximate human face proportions)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

# ==================== MOUTH LANDMARKS ====================
# MediaPipe Face Mesh mouth landmarks for mouth aspect ratio calculation
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
# Vertical mouth landmarks for MAR calculation
MOUTH_VERTICAL_TOP = [13]    # Top inner lip
MOUTH_VERTICAL_BOTTOM = [14] # Bottom inner lip
MOUTH_HORIZONTAL = [78, 308] # Left and right mouth corners

# ==================== EYE GAZE LANDMARKS ====================
# For detecting eye movement direction
RIGHT_IRIS = [468, 469, 470, 471, 472]  # Right iris landmarks (if refine_landmarks=True)
LEFT_IRIS = [473, 474, 475, 476, 477]   # Left iris landmarks (if refine_landmarks=True)

# Blink detection thresholds
EAR_THRESHOLD = 0.25  # Eye aspect ratio below this indicates closed eyes
CONSEC_FRAMES = 3     # Number of consecutive frames eyes must be closed to count as blink

# Visualization colors (BGR format)
GREEN_COLOR = (86, 241, 13)   # Eyes open / Likely Real
RED_COLOR = (30, 46, 209)     # Eyes closed / Suspicious
YELLOW_COLOR = (0, 255, 255)  # Warning / Analyzing
WHITE_COLOR = (255, 255, 255)
BLUE_COLOR = (255, 150, 50)   # Info color

# Blink detection state variables
blink_counter = 0
frame_counter = 0

# ==================== DEEPFAKE DETECTION CONFIGURATION ====================
# Rolling time window for blink analysis (in seconds)
ANALYSIS_WINDOW = 60.0  # 60 seconds rolling window

# ==========================================================================
# SUSPICION SCORE THRESHOLDS (Rule-Based Deepfake Detection)
# ==========================================================================

# 🚨 Rule 1: Very low blink rate (major signal) - adds 25 points
# Humans almost never blink this little - strong deepfake indicator
VERY_LOW_BLINK_RATE = 3     # Blinks per minute - extremely suspicious
LOW_BLINK_RATE = 8          # Blinks per minute - moderately suspicious

# 🚨 Rule 2: Long no-blink duration - adds 15 points
# Humans cannot stare that long naturally
MAX_NO_BLINK_THRESHOLD = 25.0  # seconds without blinking is suspicious

# 🚨 Rule 3: Perfect blink timing (robotic) - adds 15 points
# AI-generated blinking is often evenly spaced
ROBOTIC_CV_THRESHOLD = 0.10    # Coefficient of Variation below this = robotic

# 🚨 Rule 4: Very short blinks - adds 10 points
# Deepfakes sometimes "snap" eyes shut unnaturally
MIN_HUMAN_BLINK_DURATION = 0.1  # seconds - human blinks are 100-400ms
MAX_HUMAN_BLINK_DURATION = 0.4  # seconds

# 🚨 Rule 5: Face blur/smoothness - adds up to 15 points
# Deepfakes often have unnatural smoothness or blur on the face
BLUR_THRESHOLD_SUSPICIOUS = 80   # Low variance = too smooth (suspicious)
BLUR_THRESHOLD_VERY_SUSPICIOUS = 50  # Very low variance = very suspicious

# 🚨 Rule 6: Head pose vs eye movement mismatch - adds up to 15 points
# When head moves but eyes don't track naturally
HEAD_EYE_MISMATCH_THRESHOLD = 0.5  # Normalized mismatch threshold

# 🚨 Rule 7: Mouth movement inconsistency - adds up to 10 points
# Unnatural mouth movement patterns
MOUTH_CV_THRESHOLD_SUSPICIOUS = 0.05  # Too consistent mouth movement

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

# Head pose tracking state
head_pose_history = deque(maxlen=30)  # Store recent head poses for movement analysis
eye_position_history = deque(maxlen=30)  # Store recent eye positions

# Mouth movement tracking state
mouth_opening_history = deque(maxlen=60)  # Store recent mouth opening values

# Blur score tracking
blur_scores = deque(maxlen=30)  # Store recent blur scores


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


# ==================== HEAD POSE ESTIMATION FUNCTIONS ====================

def get_head_pose(landmarks, frame_shape):
    """
    Estimate head pose (yaw, pitch, roll) using MediaPipe face landmarks and solvePnP.

    Args:
        landmarks: MediaPipe face landmarks object
        frame_shape: Tuple of (height, width, channels) for coordinate conversion

    Returns:
        tuple: (yaw, pitch, roll) in degrees, or (0, 0, 0) if estimation fails
    """
    h, w = frame_shape[:2]

    # Extract 2D image points for the key landmarks
    image_points = np.array([
        (landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h)
        for idx in HEAD_POSE_LANDMARKS
    ], dtype=np.float64)

    # Camera matrix (approximate)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Get Euler angles from rotation matrix
        # Using the formula for extracting Euler angles (yaw, pitch, roll)
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0

        # Convert to degrees
        yaw = math.degrees(yaw)
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)

        return yaw, pitch, roll

    except Exception:
        return 0.0, 0.0, 0.0


def get_eye_center(landmarks, eye_landmarks, frame_shape):
    """
    Calculate the center position of an eye.

    Args:
        landmarks: MediaPipe face landmarks object
        eye_landmarks: List of landmark indices for the eye
        frame_shape: Tuple of (height, width) for coordinate conversion

    Returns:
        tuple: (x, y) center coordinates
    """
    h, w = frame_shape[:2]
    x_coords = []
    y_coords = []

    for idx in eye_landmarks:
        lm = landmarks.landmark[idx]
        x_coords.append(lm.x * w)
        y_coords.append(lm.y * h)

    return np.mean(x_coords), np.mean(y_coords)


def get_iris_position(landmarks, iris_landmarks, eye_landmarks, frame_shape):
    """
    Calculate the relative iris position within the eye (for gaze estimation).

    Args:
        landmarks: MediaPipe face landmarks object
        iris_landmarks: List of iris landmark indices
        eye_landmarks: List of eye landmark indices
        frame_shape: Tuple of (height, width) for coordinate conversion

    Returns:
        tuple: (x_ratio, y_ratio) where 0.5 is center, 0 is left/top, 1 is right/bottom
    """
    h, w = frame_shape[:2]

    try:
        # Get iris center
        iris_x = np.mean([landmarks.landmark[idx].x * w for idx in iris_landmarks])
        iris_y = np.mean([landmarks.landmark[idx].y * h for idx in iris_landmarks])

        # Get eye bounds
        eye_x_coords = [landmarks.landmark[idx].x * w for idx in eye_landmarks]
        eye_y_coords = [landmarks.landmark[idx].y * h for idx in eye_landmarks]

        min_x, max_x = min(eye_x_coords), max(eye_x_coords)
        min_y, max_y = min(eye_y_coords), max(eye_y_coords)

        # Calculate relative position (0-1)
        if max_x - min_x > 0 and max_y - min_y > 0:
            x_ratio = (iris_x - min_x) / (max_x - min_x)
            y_ratio = (iris_y - min_y) / (max_y - min_y)
            return x_ratio, y_ratio
    except (IndexError, ZeroDivisionError):
        pass

    return 0.5, 0.5


# ==================== MOUTH ASPECT RATIO FUNCTIONS ====================

def calculate_mouth_aspect_ratio(landmarks, frame_shape):
    """
    Calculate the Mouth Aspect Ratio (MAR) - similar to EAR but for the mouth.
    MAR = vertical_distance / horizontal_distance

    Args:
        landmarks: MediaPipe face landmarks object
        frame_shape: Tuple of (height, width) for coordinate conversion

    Returns:
        float: The mouth aspect ratio (higher = more open)
    """
    h, w = frame_shape[:2]

    def get_point(idx):
        lm = landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])

    try:
        # Get vertical mouth opening (top inner lip to bottom inner lip)
        top = get_point(MOUTH_VERTICAL_TOP[0])
        bottom = get_point(MOUTH_VERTICAL_BOTTOM[0])
        vertical_dist = np.linalg.norm(top - bottom)

        # Get horizontal mouth width (left corner to right corner)
        left = get_point(MOUTH_HORIZONTAL[0])
        right = get_point(MOUTH_HORIZONTAL[1])
        horizontal_dist = np.linalg.norm(left - right)

        if horizontal_dist > 0:
            return vertical_dist / horizontal_dist
        return 0.0
    except (IndexError, ZeroDivisionError):
        return 0.0


# ==================== BLUR DETECTION FUNCTIONS ====================

def extract_face_roi(frame, landmarks):
    """
    Extract the face region of interest (ROI) from the frame using landmarks.

    Args:
        frame: Video frame
        landmarks: MediaPipe face landmarks object

    Returns:
        numpy.ndarray: Cropped face region, or None if extraction fails
    """
    h, w = frame.shape[:2]

    try:
        # Get all landmark coordinates
        x_coords = [landmarks.landmark[i].x * w for i in range(len(landmarks.landmark))]
        y_coords = [landmarks.landmark[i].y * h for i in range(len(landmarks.landmark))]

        # Get bounding box with padding
        padding = 20
        x_min = max(0, int(min(x_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_min = max(0, int(min(y_coords)) - padding)
        y_max = min(h, int(max(y_coords)) + padding)

        if x_max > x_min and y_max > y_min:
            return frame[y_min:y_max, x_min:x_max].copy()
    except Exception:
        pass

    return None


def estimate_face_blur(face_roi):
    """
    Estimate the blur/smoothness of the face region using variance of Laplacian.

    Lower variance = more blurry/smooth (suspicious for deepfakes)
    Higher variance = sharper/more detailed (likely real)

    Args:
        face_roi: Cropped face region as numpy array

    Returns:
        tuple: (blur_score, is_suspicious) where blur_score is the variance
    """
    if face_roi is None or face_roi.size == 0:
        return 0.0, False

    try:
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Determine if suspicious
        is_suspicious = variance < BLUR_THRESHOLD_SUSPICIOUS

        return variance, is_suspicious
    except Exception:
        return 0.0, False


# ==================== MOVEMENT ANALYSIS FUNCTIONS ====================

def analyze_head_eye_mismatch(head_pose_history, eye_position_history):
    """
    Analyze if head movement and eye movement are mismatched.
    In real humans, when the head moves, eyes often compensate.
    Deepfakes may have head movement without corresponding eye tracking.

    Args:
        head_pose_history: deque of (yaw, pitch, roll) tuples
        eye_position_history: deque of (left_iris_pos, right_iris_pos) tuples

    Returns:
        tuple: (mismatch_score, is_suspicious)
    """
    if len(head_pose_history) < 10 or len(eye_position_history) < 10:
        return 0.0, False

    try:
        # Calculate head movement (yaw changes)
        head_yaws = [h[0] for h in head_pose_history]
        head_movement = np.std(head_yaws)

        # Calculate eye movement (horizontal position changes)
        eye_x_positions = []
        for eye_pos in eye_position_history:
            if eye_pos is not None and len(eye_pos) >= 2:
                # Average of left and right eye x positions
                left_x = eye_pos[0][0] if eye_pos[0] else 0.5
                right_x = eye_pos[1][0] if eye_pos[1] else 0.5
                eye_x_positions.append((left_x + right_x) / 2)

        if len(eye_x_positions) < 5:
            return 0.0, False

        eye_movement = np.std(eye_x_positions)

        # If head moves significantly but eyes don't move proportionally, it's suspicious
        # Normalize values
        head_movement_normalized = min(head_movement / 15.0, 1.0)  # 15 degrees = full movement
        eye_movement_normalized = min(eye_movement / 0.2, 1.0)    # 0.2 = full eye movement

        # Calculate mismatch (head moves but eyes don't)
        if head_movement_normalized > 0.3:  # Only check if head is moving
            mismatch = abs(head_movement_normalized - eye_movement_normalized)
            is_suspicious = mismatch > HEAD_EYE_MISMATCH_THRESHOLD
            return mismatch, is_suspicious

        return 0.0, False
    except Exception:
        return 0.0, False


def analyze_mouth_consistency(mouth_opening_history):
    """
    Analyze mouth movement consistency.
    Too consistent (robotic) or too erratic mouth movement can be suspicious.

    Args:
        mouth_opening_history: deque of mouth aspect ratio values

    Returns:
        tuple: (consistency_score, is_suspicious)
    """
    if len(mouth_opening_history) < 20:
        return 0.0, False

    try:
        values = list(mouth_opening_history)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val > 0:
            cv = std_val / mean_val
            # Too consistent is suspicious (robotic movement)
            is_suspicious = cv < MOUTH_CV_THRESHOLD_SUSPICIOUS and mean_val > 0.1
            return cv, is_suspicious

        return 0.0, False
    except Exception:
        return 0.0, False


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


def analyze_blink_behavior(current_time, blur_score=None, head_eye_mismatch=None, mouth_consistency=None):
    """
    Analyze blink behavior and determine if video is likely real or deepfake.

    Uses SUSPICION SCORE rule-based decision logic:
    🚨 Rule 1: Very low blink rate (<3/min) -> +25 points
    🚨 Rule 2: Long no-blink duration (>25s) -> +15 points
    🚨 Rule 3: Perfect blink timing (robotic CV) -> +15 points
    🚨 Rule 4: Very short blinks (<100ms) -> +10 points
    🚨 Rule 5: Face blur/smoothness -> up to +15 points
    🚨 Rule 6: Head-eye mismatch -> up to +10 points
    🚨 Rule 7: Mouth movement inconsistency -> up to +10 points

    Args:
        current_time: Current timestamp
        blur_score: Face blur variance (lower = more suspicious)
        head_eye_mismatch: Head vs eye movement mismatch score
        mouth_consistency: Mouth movement consistency (CV)

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
        'avg_blink_duration': avg_blink_duration,
        'blur_score': blur_score,
        'head_eye_mismatch': head_eye_mismatch,
        'mouth_consistency': mouth_consistency
    }

    # Not enough data yet
    if elapsed_time < MIN_ANALYSIS_TIME:
        return "Analyzing...", 0.0, ["Collecting data..."], metrics

    # ==========================================================================
    # SUSPICION SCORE CALCULATION (0-100 scale)
    # ==========================================================================
    suspicion_score = 0
    reasons = []

    # 🚨 Rule 1: Very low blink rate (major signal) - adds up to 25 points
    # Humans almost never blink this little - strong deepfake indicator
    if blink_rate < VERY_LOW_BLINK_RATE:
        suspicion_score += 25
        reasons.append(f"🚨 Very low blink rate: {blink_rate:.1f}/min (<{VERY_LOW_BLINK_RATE})")
    elif blink_rate < LOW_BLINK_RATE:
        suspicion_score += 12
        reasons.append(f"⚠️ Low blink rate: {blink_rate:.1f}/min (<{LOW_BLINK_RATE})")

    # 🚨 Rule 2: Long no-blink duration - adds up to 15 points
    # Humans cannot stare that long naturally
    effective_max_no_blink = max(max_no_blink_duration, current_no_blink)
    if effective_max_no_blink > MAX_NO_BLINK_THRESHOLD:
        suspicion_score += 15
        reasons.append(f"🚨 Long stare: {effective_max_no_blink:.1f}s without blinking")
    elif effective_max_no_blink > MAX_NO_BLINK_THRESHOLD * 0.6:
        suspicion_score += 8
        reasons.append(f"⚠️ Extended stare: {effective_max_no_blink:.1f}s")

    # 🚨 Rule 3: Perfect blink timing (robotic) - adds up to 15 points
    # AI-generated blinking is often evenly spaced
    if cv is not None and cv < ROBOTIC_CV_THRESHOLD:
        suspicion_score += 15
        reasons.append(f"🚨 Robotic blink pattern (CV={cv:.3f})")
    elif cv is not None and cv < 0.15:
        suspicion_score += 8
        reasons.append(f"⚠️ Suspiciously regular blinks (CV={cv:.3f})")

    # 🚨 Rule 4: Very short blinks - adds up to 10 points
    # Deepfakes sometimes "snap" eyes shut unnaturally
    if avg_blink_duration is not None:
        if avg_blink_duration < MIN_HUMAN_BLINK_DURATION:
            suspicion_score += 10
            reasons.append(f"🚨 Unnatural blink speed: {avg_blink_duration*1000:.0f}ms")
        elif avg_blink_duration > MAX_HUMAN_BLINK_DURATION:
            suspicion_score += 5
            reasons.append(f"⚠️ Slow blinks: {avg_blink_duration*1000:.0f}ms")

    # 🚨 Rule 5: Face blur/smoothness - adds up to 15 points
    if blur_score is not None:
        if blur_score < BLUR_THRESHOLD_VERY_SUSPICIOUS:
            suspicion_score += 15
            reasons.append(f"🚨 Face too smooth: blur={blur_score:.0f}")
        elif blur_score < BLUR_THRESHOLD_SUSPICIOUS:
            suspicion_score += 8
            reasons.append(f"⚠️ Face smoothness: blur={blur_score:.0f}")

    # 🚨 Rule 6: Head-eye movement mismatch - adds up to 10 points
    if head_eye_mismatch is not None and head_eye_mismatch[1]:  # (score, is_suspicious)
        mismatch_score = head_eye_mismatch[0]
        if mismatch_score > HEAD_EYE_MISMATCH_THRESHOLD * 1.5:
            suspicion_score += 10
            reasons.append(f"🚨 Head-eye mismatch: {mismatch_score:.2f}")
        elif mismatch_score > HEAD_EYE_MISMATCH_THRESHOLD:
            suspicion_score += 5
            reasons.append(f"⚠️ Slight head-eye mismatch")

    # 🚨 Rule 7: Mouth movement inconsistency - adds up to 10 points
    if mouth_consistency is not None and mouth_consistency[1]:  # (cv, is_suspicious)
        mouth_cv = mouth_consistency[0]
        suspicion_score += 10
        reasons.append(f"🚨 Robotic mouth movement (CV={mouth_cv:.3f})")

    # ==========================================================================
    # FINAL CONFIDENCE MAPPING
    # ==========================================================================
    if suspicion_score >= 70:
        confidence = min(0.95, 0.70 + (suspicion_score - 70) * 0.01)
        result = "LIKELY DEEPFAKE"
    elif suspicion_score >= 40:
        confidence = 0.50 + (suspicion_score - 40) * 0.007
        result = "SUSPICIOUS"
    else:
        # Low suspicion = likely real
        confidence = max(0.30, (100 - suspicion_score) / 100)
        result = "LIKELY REAL"
        if len(reasons) == 0:
            reasons.append("✅ Normal blink patterns detected")
            reasons.append("✅ Natural facial movements")

    # Add suspicion score to metrics
    metrics['suspicion_score'] = suspicion_score

    return result, confidence, reasons, metrics


def draw_detection_overlay(frame, result, confidence, reasons, metrics, head_pose=None):
    """
    Draw the deepfake detection results on the frame.

    Args:
        frame: Video frame to draw on
        result: Detection result string
        confidence: Confidence score (0-1)
        reasons: List of reason strings
        metrics: Dictionary of analysis metrics
        head_pose: Tuple of (yaw, pitch, roll) in degrees
    """
    h, w = frame.shape[:2]
    suspicion_score = metrics.get('suspicion_score', 0)

    # Choose colors based on result
    if "DEEPFAKE" in result:
        result_color = RED_COLOR
        bg_color = (0, 0, 120)
    elif "SUSPICIOUS" in result:
        result_color = RED_COLOR
        bg_color = (0, 0, 100)
    elif "REAL" in result:
        result_color = GREEN_COLOR
        bg_color = (0, 100, 0)
    else:
        result_color = YELLOW_COLOR
        bg_color = (100, 100, 0)

    # Draw semi-transparent background panel (right side)
    overlay = frame.copy()
    panel_height = 280 + len(reasons) * 20
    cv2.rectangle(overlay, (w - 380, 10), (w - 10, min(panel_height, h - 10)), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw title
    cv2.putText(frame, "DEEPFAKE DETECTION", (w - 370, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE_COLOR, 2)

    # Draw detection result
    cv2.putText(frame, result, (w - 370, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, result_color, 2)

    # Draw suspicion score
    cv2.putText(frame, f"Suspicion Score: {suspicion_score}/100", (w - 370, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE_COLOR, 1)

    # Draw suspicion score bar
    bar_width = int(300 * (suspicion_score / 100))
    if suspicion_score < 40:
        bar_color = GREEN_COLOR
    elif suspicion_score < 70:
        bar_color = YELLOW_COLOR
    else:
        bar_color = RED_COLOR
    cv2.rectangle(frame, (w - 370, 98), (w - 370 + bar_width, 112), bar_color, cv2.FILLED)
    cv2.rectangle(frame, (w - 370, 98), (w - 70, 112), WHITE_COLOR, 1)

    # Draw metrics section header
    y_pos = 135
    cv2.putText(frame, "--- ANALYSIS METRICS ---", (w - 370, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE_COLOR, 1)

    # Blink metrics
    y_pos += 22
    blink_rate = metrics.get('blink_rate', 0)
    elapsed = metrics.get('elapsed_time', 0)
    cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min", (w - 370, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    y_pos += 18
    avg_duration = metrics.get('avg_blink_duration')
    if avg_duration:
        cv2.putText(frame, f"Avg Blink Duration: {avg_duration*1000:.0f}ms", (w - 370, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)
    else:
        cv2.putText(frame, "Avg Blink Duration: --", (w - 370, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    y_pos += 18
    max_no_blink = metrics.get('max_no_blink', 0)
    cv2.putText(frame, f"Max No-Blink: {max_no_blink:.1f}s", (w - 370, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    # Blur score
    y_pos += 18
    blur = metrics.get('blur_score')
    if blur is not None:
        blur_text = f"Face Blur Score: {blur:.0f}"
        blur_color = GREEN_COLOR if blur > BLUR_THRESHOLD_SUSPICIOUS else (YELLOW_COLOR if blur > BLUR_THRESHOLD_VERY_SUSPICIOUS else RED_COLOR)
        cv2.putText(frame, blur_text, (w - 370, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, blur_color, 1)
    else:
        cv2.putText(frame, "Face Blur Score: --", (w - 370, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    # Head pose
    y_pos += 18
    if head_pose:
        yaw, pitch, roll = head_pose
        cv2.putText(frame, f"Head Pose: Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}", (w - 370, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)
    else:
        cv2.putText(frame, "Head Pose: --", (w - 370, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    # Session time
    y_pos += 18
    cv2.putText(frame, f"Analysis Time: {elapsed:.0f}s", (w - 370, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    # Draw reasons section
    y_pos += 25
    cv2.putText(frame, "--- DETECTION REASONS ---", (w - 370, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE_COLOR, 1)

    y_pos += 20
    for reason in reasons[:5]:  # Show max 5 reasons
        # Truncate long reasons
        display_reason = reason[:42] + "..." if len(reason) > 42 else reason
        cv2.putText(frame, display_reason, (w - 370, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE_COLOR, 1)
        y_pos += 18


def draw_left_panel(frame, blink_counter, ear, mar, head_pose, blur_score):
    """
    Draw the left side information panel with real-time metrics.

    Args:
        frame: Video frame to draw on
        blink_counter: Total blink count
        ear: Current Eye Aspect Ratio
        mar: Current Mouth Aspect Ratio
        head_pose: Tuple of (yaw, pitch, roll) in degrees
        blur_score: Face blur variance
    """
    h, w = frame.shape[:2]

    # Draw semi-transparent background panel (left side)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (220, 180), (50, 50, 50), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "REAL-TIME METRICS", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE_COLOR, 1)

    # Blink count
    cv2.putText(frame, f"Blinks: {blink_counter}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE_COLOR, 1)

    # EAR
    ear_color = RED_COLOR if ear < EAR_THRESHOLD else GREEN_COLOR
    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 1)

    # MAR
    cv2.putText(frame, f"MAR: {mar:.3f}", (20, 101),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE_COLOR, 1)

    # Head pose
    if head_pose:
        yaw, pitch, roll = head_pose
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (100, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)
        cv2.putText(frame, f"Roll: {roll:.1f}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    # Blur score
    if blur_score is not None:
        cv2.putText(frame, f"Blur: {blur_score:.0f}", (100, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE_COLOR, 1)

    # Status indicator
    cv2.putText(frame, "LIVE", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN_COLOR, 1)


# ==================== MAIN VIDEO PROCESSING LOOP ====================

def main(video_source=0):
    """
    Main function to run the deepfake detection system.

    Args:
        video_source: Video source (0 for webcam, or path to video file)
    """
    global blink_counter, frame_counter, blink_start_time, last_blink_time
    global max_no_blink_duration, analysis_start_time
    global blink_timestamps, blink_durations
    global head_pose_history, eye_position_history, mouth_opening_history, blur_scores

    # Reset state variables
    blink_counter = 0
    frame_counter = 0
    blink_start_time = None
    last_blink_time = None
    max_no_blink_duration = 0.0
    analysis_start_time = None
    blink_timestamps.clear()
    blink_durations.clear()
    head_pose_history.clear()
    eye_position_history.clear()
    mouth_opening_history.clear()
    blur_scores.clear()

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Video source: {video_source}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.1f}")
    print("Press 'ESC' to exit, 'R' to reset analysis")
    print("-" * 50)

    # Initialize timing
    analysis_start_time = time.time()

    # Variables for tracking
    current_blur_score = None
    current_head_pose = None
    current_mar = 0.0

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # If video file ends, break
                if video_source != 0:
                    break
                continue

            current_time = time.time()

            # Flip only for webcam (mirror effect)
            if video_source == 0:
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
                            color=(180, 180, 180), thickness=1, circle_radius=1
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

                    # ========== HEAD POSE ESTIMATION ==========
                    current_head_pose = get_head_pose(face_landmarks, frame.shape)
                    head_pose_history.append(current_head_pose)

                    # ========== EYE TRACKING (IRIS POSITION) ==========
                    try:
                        left_iris_pos = get_iris_position(
                            face_landmarks, LEFT_IRIS, LEFT_EYE, frame.shape)
                        right_iris_pos = get_iris_position(
                            face_landmarks, RIGHT_IRIS, RIGHT_EYE, frame.shape)
                        eye_position_history.append((left_iris_pos, right_iris_pos))
                    except Exception:
                        eye_position_history.append(None)

                    # ========== MOUTH ASPECT RATIO ==========
                    current_mar = calculate_mouth_aspect_ratio(face_landmarks, frame.shape)
                    mouth_opening_history.append(current_mar)

                    # ========== FACE BLUR DETECTION ==========
                    face_roi = extract_face_roi(frame, face_landmarks)
                    if face_roi is not None:
                        current_blur_score, _ = estimate_face_blur(face_roi)
                        blur_scores.append(current_blur_score)

                    # ========== ANALYZE ADDITIONAL SIGNALS ==========
                    # Analyze head-eye mismatch
                    head_eye_mismatch = analyze_head_eye_mismatch(
                        head_pose_history, eye_position_history)

                    # Analyze mouth consistency
                    mouth_consistency = analyze_mouth_consistency(mouth_opening_history)

                    # Get average blur score for stability
                    avg_blur = np.mean(list(blur_scores)) if blur_scores else None

                    # ========== DEEPFAKE DETECTION ANALYSIS ==========
                    result, confidence, reasons, metrics = analyze_blink_behavior(
                        current_time,
                        blur_score=avg_blur,
                        head_eye_mismatch=head_eye_mismatch,
                        mouth_consistency=mouth_consistency
                    )

                    # ========== DRAW OVERLAYS ==========
                    # Draw left panel with real-time metrics
                    draw_left_panel(frame, blink_counter, avg_ear, current_mar,
                                   current_head_pose, current_blur_score)

                    # Draw right panel with detection results
                    draw_detection_overlay(frame, result, confidence, reasons, metrics,
                                          head_pose=current_head_pose)

                    # Visual feedback when blink is detected
                    if blink_detected:
                        cv2.putText(frame, "BLINK!", (frame.shape[1]//2 - 50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN_COLOR, 2)

            else:
                # No face detected - show warning
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (300, 60), (0, 0, 100), cv2.FILLED)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "No face detected", (20, 42),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW_COLOR, 2)

            cv2.imshow("Deepfake Video Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('r') or key == ord('R'):
                # Reset analysis
                print("Resetting analysis...")
                blink_counter = 0
                frame_counter = 0
                blink_start_time = None
                last_blink_time = None
                max_no_blink_duration = 0.0
                analysis_start_time = time.time()
                blink_timestamps.clear()
                blink_durations.clear()
                head_pose_history.clear()
                eye_position_history.clear()
                mouth_opening_history.clear()
                blur_scores.clear()

    cap.release()
    cv2.destroyAllWindows()

    # Print final analysis summary
    print("\n" + "="*60)
    print("          DEEPFAKE DETECTION SESSION SUMMARY")
    print("="*60)
    print(f"Total blinks detected: {blink_counter}")
    session_duration = time.time() - analysis_start_time if analysis_start_time else 0
    print(f"Session duration: {session_duration:.1f} seconds")
    print(f"Max no-blink duration: {max_no_blink_duration:.1f} seconds")

    # Get average blur score
    avg_blur = np.mean(list(blur_scores)) if blur_scores else None
    head_eye_mismatch = analyze_head_eye_mismatch(head_pose_history, eye_position_history)
    mouth_consistency = analyze_mouth_consistency(mouth_opening_history)

    final_result, final_conf, final_reasons, final_metrics = analyze_blink_behavior(
        time.time(),
        blur_score=avg_blur,
        head_eye_mismatch=head_eye_mismatch,
        mouth_consistency=mouth_consistency
    )

    print("-"*60)
    print(f"🔍 FINAL RESULT: {final_result}")
    print(f"📊 Suspicion Score: {final_metrics.get('suspicion_score', 0)}/100")
    print(f"📈 Confidence: {final_conf*100:.1f}%")
    print(f"👁️  Blink Rate: {final_metrics['blink_rate']:.1f} blinks/min")
    if final_metrics['cv'] is not None:
        print(f"⏱️  Interval Regularity (CV): {final_metrics['cv']:.3f}")
    if final_metrics.get('avg_blink_duration') is not None:
        print(f"⚡ Avg Blink Duration: {final_metrics['avg_blink_duration']*1000:.0f}ms")
    if avg_blur is not None:
        print(f"🔍 Avg Face Blur Score: {avg_blur:.0f}")
    print("-"*60)
    print("Detection Reasons:")
    for reason in final_reasons:
        print(f"  {reason}")
    print("="*60)


# Run the main function when script is executed directly
if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Processing video file: {video_path}")
        main(video_source=video_path)
    else:
        print("Starting webcam capture (use 'python face_mesh.py <video_path>' for video file)")
        main(video_source=0)

