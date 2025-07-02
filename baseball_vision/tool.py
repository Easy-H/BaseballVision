import mediapipe as mp
import baseball_vision.angle_calc as ac

def get_joints(landmarks):
     return { "R_shoulder": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
                "R_elbow": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
                "R_wrist": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
                "R_hip": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP],
                "R_knee": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE],
                "R_ankle": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE],
                "L_shoulder": landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
                "L_elbow": landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
                "L_wrist": landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST],
                "L_hip": landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP],
                "L_knee": landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE],
                "L_ankle": landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE],
                "R_pinky_tip": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX],
                "L_pinky_tip": landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]}

def pitcher_tool(landmarks):

    joints = get_joints(landmarks)
    
    # --- 각도 계산 ---
    # 1. 오른쪽 팔꿈치 각도
    angle_R_elbow = ac.calculate_angle_3(joints["R_shoulder"], joints["R_elbow"], joints["R_wrist"])
    angle_L_elbow = ac.calculate_angle_3(joints["L_shoulder"], joints["L_elbow"], joints["L_wrist"])

    # 2. 오른쪽 어깨 각도 (몸통-어깨-팔꿈치"])
    angle_R_shoulder = ac.calculate_angle_3(joints["L_shoulder"], joints["R_shoulder"], joints["R_elbow"])
    angle_L_shoulder = ac.calculate_angle_3(joints["R_shoulder"], joints["L_shoulder"], joints["L_elbow"])

    # 3. 오른쪽 골반 각도 (몸통-골반-무릎"])
    angle_body_twist = ac.calculate_angle_4(joints["R_shoulder"], joints["L_shoulder"], joints["R_hip"], joints["L_hip"])

    # 4. 오른쪽 무릎 각도
    angle_R_knee = ac.calculate_angle_3(joints["R_hip"], joints["R_knee"], joints["R_ankle"])
    angle_L_knee = ac.calculate_angle_3(joints["L_hip"], joints["L_knee"], joints["L_ankle"])

    return [ ["R Elbow", str(round(angle_R_elbow, 2))],
          ["L Elbow", str(round(angle_L_elbow, 2))],
          ["R Shoulder", str(round(angle_R_shoulder, 2))],
          ["L Shoulder", str(round(angle_L_shoulder, 2))],
          ["Body Twist", str(round(angle_body_twist, 2))],
          ["R Knee", str(round(angle_R_knee, 2))],
          ["L Knee", str(round(angle_L_knee, 2))]] 

def batter_tool(landmarks):
    joints = get_joints(landmarks)
    
    # --- 각도 계산 ---
    # 1. 팔꿈치 각도
    angle_R_elbow = ac.calculate_angle_3(joints["R_shoulder"], joints["R_elbow"], joints["R_wrist"])
    angle_L_elbow = ac.calculate_angle_3(joints["L_shoulder"], joints["L_elbow"], joints["L_wrist"])

    # 2. 어깨 각도
    angle_R_shoulder = ac.calculate_angle_3(joints["L_shoulder"], joints["R_shoulder"], joints["R_elbow"])
    angle_L_shoulder = ac.calculate_angle_3(joints["R_shoulder"], joints["L_shoulder"], joints["L_elbow"])

    # 3. 꼬임
    angle_body_twist = ac.calculate_angle_4(joints["R_shoulder"], joints["L_shoulder"], joints["R_hip"], joints["L_hip"])

    # 4. 무릎 각도
    angle_R_knee = ac.calculate_angle_3(joints["R_hip"], joints["R_knee"], joints["R_ankle"])
    angle_L_knee = ac.calculate_angle_3(joints["L_hip"], joints["L_knee"], joints["L_ankle"])
    
    angle_R_wrist = ac.calculate_angle_3(joints["R_elbow"], joints["R_wrist"], joints["R_pinky_tip"])
    angle_L_wrist = ac.calculate_angle_3(joints["L_elbow"], joints["L_wrist"], joints["L_pinky_tip"])

    return [ ["R Elbow", str(round(angle_R_elbow, 2))],
          ["L Elbow", str(round(angle_L_elbow, 2))],
          ["R Shoulder", str(round(angle_R_shoulder, 2))],
          ["L Shoulder", str(round(angle_L_shoulder, 2))],
          ["Body Twist", str(round(angle_body_twist, 2))],
          ["R Knee", str(round(angle_R_knee, 2))],
          ["L Knee", str(round(angle_L_knee, 2))],
          ["R wrist", str(round(angle_R_wrist, 2))],
          ["L wrist", str(round(angle_L_wrist, 2))]]  