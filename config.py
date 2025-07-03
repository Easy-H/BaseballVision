import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

MODEL_COMPLEXITY = 2
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3
MIN_DRAW_VISIBILITY = 0.2
PROCESS_NOISE_STD = 0.001
MEASUREMENT_NOISE_STD = 0.05
APPLY_KALMAN_FILTER = False
OUTPUT_DIR = 'result_data'

# MediaPipe Landmark Names for C3D Marker Mapping
MEDIAPIPE_LANDMARK_NAMES = {
    mp_pose.PoseLandmark.NOSE.value: 'NOSE',
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value: 'L_EYE_INNER',
    mp_pose.PoseLandmark.LEFT_EYE.value: 'L_EYE',
    mp_pose.PoseLandmark.LEFT_EYE_OUTER.value: 'L_EYE_OUTER',
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value: 'R_EYE_INNER',
    mp_pose.PoseLandmark.RIGHT_EYE.value: 'R_EYE',
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value: 'R_EYE_OUTER',
    mp_pose.PoseLandmark.LEFT_EAR.value: 'L_EAR',
    mp_pose.PoseLandmark.RIGHT_EAR.value: 'R_EAR',
    mp_pose.PoseLandmark.MOUTH_LEFT.value: 'L_MOUTH',
    mp_pose.PoseLandmark.MOUTH_RIGHT.value: 'R_MOUTH',
    mp_pose.PoseLandmark.LEFT_SHOULDER.value: 'L_SHOULDER',
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value: 'R_SHOULDER',
    mp_pose.PoseLandmark.LEFT_ELBOW.value: 'L_ELBOW',
    mp_pose.PoseLandmark.RIGHT_ELBOW.value: 'R_ELBOW',
    mp_pose.PoseLandmark.LEFT_WRIST.value: 'L_WRIST',
    mp_pose.PoseLandmark.RIGHT_WRIST.value: 'R_WRIST',
    mp_pose.PoseLandmark.LEFT_PINKY.value: 'L_PINKY',
    mp_pose.PoseLandmark.RIGHT_PINKY.value: 'R_PINKY',
    mp_pose.PoseLandmark.LEFT_INDEX.value: 'L_INDEX',
    mp_pose.PoseLandmark.RIGHT_INDEX.value: 'R_INDEX',
    mp_pose.PoseLandmark.LEFT_THUMB.value: 'L_THUMB',
    mp_pose.PoseLandmark.RIGHT_THUMB.value: 'R_THUMB',
    mp_pose.PoseLandmark.LEFT_HIP.value: 'L_HIP',
    mp_pose.PoseLandmark.RIGHT_HIP.value: 'R_HIP',
    mp_pose.PoseLandmark.LEFT_KNEE.value: 'L_KNEE',
    mp_pose.PoseLandmark.RIGHT_KNEE.value: 'R_KNEE',
    mp_pose.PoseLandmark.LEFT_ANKLE.value: 'L_ANKLE',
    mp_pose.PoseLandmark.RIGHT_ANKLE.value: 'R_ANKLE',
    mp_pose.PoseLandmark.LEFT_HEEL.value: 'L_HEEL',
    mp_pose.PoseLandmark.RIGHT_HEEL.value: 'R_HEEL',
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value: 'L_FOOT_INDEX',
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value: 'R_FOOT_INDEX',
}

ALL_POSE_LANDMARK_INDICES = list(range(len(mp_pose.PoseLandmark)))
ALL_POSE_MARKER_NAMES = [MEDIAPIPE_LANDMARK_NAMES.get(i, f'UNKNOWN_{i}') for i in ALL_POSE_LANDMARK_INDICES]

# --- Color Definitions for Drawing ---
COLOR_LEFT_ARM = (255, 0, 0)     # Blue (Left Arm)
COLOR_RIGHT_ARM = (0, 0, 255)    # Red (Right Arm)
COLOR_LEFT_LEG = (255, 255, 0)   # Cyan (Left Leg)
COLOR_RIGHT_LEG = (0, 255, 255)  # Yellow (Right Leg)
COLOR_TORSO = (0, 255, 0)        # Green (Torso)
COLOR_HEAD_NECK = (255, 255, 255) # White (Head/Neck)

# Map connections to their respective colors
CONNECTIONS_COLORS = {
    # Arms
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value): COLOR_LEFT_ARM,
    (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value): COLOR_LEFT_ARM,
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value): COLOR_RIGHT_ARM,
    (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value): COLOR_RIGHT_ARM,
    
    # Legs
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value): COLOR_LEFT_LEG,
    (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value): COLOR_LEFT_LEG,
    (mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_HEEL.value): COLOR_LEFT_LEG,
    (mp_pose.PoseLandmark.LEFT_HEEL.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value): COLOR_LEFT_LEG,
    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value): COLOR_RIGHT_LEG,
    (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value): COLOR_RIGHT_LEG,
    (mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_HEEL.value): COLOR_RIGHT_LEG,
    (mp_pose.PoseLandmark.RIGHT_HEEL.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value): COLOR_RIGHT_LEG,

    # Torso
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value): COLOR_TORSO,
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value): COLOR_TORSO,
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value): COLOR_TORSO,
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value): COLOR_TORSO,
    
    # Head/Neck (Nose to shoulders to represent neck for simplified face)
    (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value): COLOR_HEAD_NECK, 
    (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value): COLOR_HEAD_NECK,
}