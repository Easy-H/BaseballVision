MODEL_COMPLEXITY = 2
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3
MIN_DRAW_VISIBILITY = 0.2
PROCESS_NOISE_STD = 0.001
MEASUREMENT_NOISE_STD = 0.05
APPLY_KALMAN_FILTER = True
OUTPUT_DIR = 'result_data'

MODEL_CONFIG = {
    "MODEL_COMPLEXITY" : 2,
    "MIN_DETECTION_CONFIDENCE" : 0.3,
    "MIN_TRACKING_CONFIDENCE" : 0.3,
    "MIN_DRAW_VISIBILITY" : 0.2,
    "PROCESS_NOISE_STD" : 0.001,
    "MEASUREMENT_NOISE_STD" : 0.05,
    "APPLY_KALMAN_FILTER" : True,
    "OUTPUT_DIR" : 'result_data'
}

JOINTS = [
    "L_SHOULDER","L_ELBOW","L_WRIST",
    "R_SHOULDER","R_ELBOW","R_WRIST",
    "L_HIP","L_KNEE","L_ANKLE","L_HEEL","L_FOOT_INDEX",
    "R_HIP","R_KNEE","R_ANKLE","R_HEEL","R_FOOT_INDEX",
    "NOSE"
]

POSE_CONNECTIONS = [
    ("L_SHOULDER","L_ELBOW"),
    ("L_ELBOW","L_WRIST"),
    ("R_SHOULDER","R_ELBOW"),
    ("R_ELBOW","R_WRIST"),
    ("L_HIP","L_KNEE"),
    ("L_KNEE","L_ANKLE"),
    ("L_ANKLE","L_HEEL"),
    ("L_HEEL","L_FOOT_INDEX"),
    ("R_HIP","R_KNEE"),
    ("R_KNEE","R_ANKLE"),
    ("R_ANKLE","R_HEEL"),
    ("R_HEEL","R_FOOT_INDEX"),
    ("L_SHOULDER","R_SHOULDER"),
    ("L_HIP","R_HIP"),
    ("L_SHOULDER","L_HIP"),
    ("R_SHOULDER","R_HIP")
]

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
    ("L_SHOULDER","L_ELBOW"): COLOR_LEFT_ARM,
    ("L_ELBOW","L_WRIST"): COLOR_LEFT_ARM,
    ("R_SHOULDER","R_ELBOW"): COLOR_RIGHT_ARM,
    ("R_ELBOW","R_WRIST"): COLOR_RIGHT_ARM,
    
    # Legs
    ("L_HIP","L_KNEE"): COLOR_LEFT_LEG,
    ("L_KNEE","L_ANKLE"): COLOR_LEFT_LEG,
    ("L_ANKLE","L_HEEL"): COLOR_LEFT_LEG,
    ("L_HEEL","L_FOOT_INDEX"): COLOR_LEFT_LEG,
    ("R_HIP","R_KNEE"): COLOR_RIGHT_LEG,
    ("R_KNEE","R_ANKLE"): COLOR_RIGHT_LEG,
    ("R_ANKLE","R_HEEL"): COLOR_RIGHT_LEG,
    ("R_HEEL","R_FOOT_INDEX"): COLOR_RIGHT_LEG,

    # Torso
    ("L_SHOULDER","R_SHOULDER"): COLOR_TORSO,
    ("L_HIP","R_HIP"): COLOR_TORSO,
    ("L_SHOULDER","L_HIP"): COLOR_TORSO,
    ("R_SHOULDER","R_HIP"): COLOR_TORSO,
    
    # Head/Neck (Nose to shoulders to represent neck for simplified face")
    ("NOSE","L_SHOULDER"): COLOR_HEAD_NECK,
    ("NOSE","R_SHOULDER"): COLOR_HEAD_NECK,

}