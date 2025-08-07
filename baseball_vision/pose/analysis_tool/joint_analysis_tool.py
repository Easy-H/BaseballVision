from .analysis_tool import AnalysisTool
from ..processed_data import ProcessedData

import numpy as np
import pandas as pd

joint_calc_parameter = {
    "R_ELBOW": ["R_SHOULDER", "R_ELBOW", "R_WRIST"],
    "L_ELBOW": ["L_SHOULDER", "L_ELBOW", "L_WRIST"],
    "R_SHOULDER": ["L_SHOULDER", "R_SHOULDER", "R_ELBOW"],
    "L_SHOULDER": ["R_SHOULDER", "L_SHOULDER", "L_ELBOW"],
    "R_KNEE": ["R_HIP", "R_KNEE", "R_ANKLE"],
    "L_KNEE": ["L_HIP", "L_KNEE", "L_ANKLE"],
    "R_WRIST": ["R_ELBOW", "R_WRIST", "R_PINKY"],
    "L_WRIST": ["L_ELBOW", "L_WRIST", "L_PINKY"]
        
}

''''''

class JointAnalysisTool(AnalysisTool):
    def __init__(self): 
        pass
    def calc(self, data:ProcessedData):

        landmarks_3d_list = data.get_landmarks_3d()
        
        results = [
            self.calc_joints(landmarks_3d) if landmarks_3d is not None else {}
            for landmarks_3d in landmarks_3d_list
        ]
        
        df = pd.DataFrame(results)

        df = df.infer_objects(copy=False)
        df_interpolated = df.interpolate(method='linear', axis=0)

        df_final = df_interpolated.ffill().bfill()

        return df_final
    
    def calc_joints(self, joints):
        # --- 각도 계산 ---
        ret = {}

        for name, value in joint_calc_parameter.items():
            ret[name] = self._calc_joint(joints, value)

        ret["SHOULDER"] = self._calc_joint4(joints,
                                            ["L_SHOULDER", "R_SHOULDER"])
        ret["PELVIS"] = self._calc_joint4(joints, ["L_HIP", "R_HIP"])
            
        if self.check_data_exist(ret, ["SHOULDER", "PELVIS"]):
            angle_body_twist = ret["SHOULDER"] - ret["PELVIS"]
            ret["TWIST"] = round((angle_body_twist + 180) % 360 - 180, 2)
        
        return ret
    
    def _calc_joint(self, joints, name_list):
        if not self.check_data_exist(joints, name_list):
            return None
        
        return calculate_angle_3(
            joints[name_list[0]], joints[name_list[1]],
            joints[name_list[2]])

    def _calc_joint4(self, joints, name_list):
        if not self.check_data_exist(joints, name_list):
            return None
        
        return calculate_angle_4(
            joints[name_list[0]], joints[name_list[1]],
            np.array([0, 0, 0]), np.array([10, 0, 0]))

    def items(self):
        return ["R_ELBOW", "L_ELBOW",
                "R_SHOULDER", "L_SHOULDER",
                "SHOULDER", "PELVIS", "TWIST",
                "R_KNEE", "L_KNEE",
                "R_WRIST", "L_WRIST"]

def calculate_angle(vec1, vec2, r=2):
    # 코사인 값 계산
    # np.dot(ba, bc)는 내적, np.linalg.norm()은 벡터의 크기
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 아크코사인으로 라디안 각도 계산
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # clip으로 부동 소수점 오차 방지

    # 라디안을 도로 변환
    angle_degrees = np.degrees(angle_radians)

    return round(angle_degrees, r)

def calculate_angle_3(a, b, c, r=2):
    return calculate_angle(a - b, c - b, r)

def calculate_angle_4(a, b, c, d, r=2):
    v1 = np.array([a[0] - b[0], a[2] - b[2]])
    v2 = np.array([c[0] - d[0], c[2] - d[2]])
    
    # 아크코사인으로 라디안 각도 계산
    angle_radians = np.arctan2(v1[0]*v2[1] - v1[1]*v2[0], v1[0]*v2[0] + v1[1]*v2[1])
    angle_degrees = np.degrees(angle_radians)

    return round(angle_degrees, r)