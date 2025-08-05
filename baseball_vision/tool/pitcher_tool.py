import numpy as np
from . import angle_calc as ac
from .tool import AnalysisTool

class PitcherAnalysisTool(AnalysisTool):
    # [수정 사항 16] __init__ 메서드: 상위 클래스의 초기화를 호출하고, focus_label 기본값 설정
    def __init__(self, label:list=[]): 
        super().__init__(label) # AnalysisTool의 __init__ 호출
        if not label: # 전달된 label이 없으면 기본 focus_label 설정
            self.focus_label = ["R Elbow", "R Shoulder", "Shoulder", "Pelvis", "Body Twist", "R Knee"] 

    def calc_joints(self, joints):
        # --- 각도 계산 ---
        ret = {}
        # 1. 오른쪽 팔꿈치 각도
        if self.joint_exist_check(joints, ["R_SHOULDER", "R_ELBOW", "R_WRIST"]):
            ret["R Elbow"] = ac.calculate_angle_3(
                joints["R_SHOULDER"],
                joints["R_ELBOW"],
                joints["R_WRIST"])
        
        if self.joint_exist_check(joints, ["L_SHOULDER", "L_ELBOW", "L_WRIST"]):
            ret["L Elbow"] = ac.calculate_angle_3(
                joints["L_SHOULDER"],
                joints["L_ELBOW"],
                joints["L_WRIST"])
    
        if self.joint_exist_check(joints, ["L_SHOULDER", "R_SHOULDER", "R_ELBOW"]):
            ret["R Shoulder"] = ac.calculate_angle_3(
                joints["L_SHOULDER"],
                joints["R_SHOULDER"],
                joints["R_ELBOW"])
        
        if self.joint_exist_check(joints, ["R_SHOULDER", "L_SHOULDER", "L_ELBOW"]):
            ret["L Shoulder"] = ac.calculate_angle_3(
                joints["R_SHOULDER"],
                joints["L_SHOULDER"],
                joints["L_ELBOW"])
        
        if self.joint_exist_check(joints, ["L_SHOULDER", "R_SHOULDER"]):
            ret["Shoulder"] = ac.calculate_angle_4(
                joints["R_SHOULDER"],
                joints["L_SHOULDER"],
                np.array([0, 0, 0]), np.array([10, 0, 0]))

        if self.joint_exist_check(joints, ["L_HIP", "R_HIP"]):
            ret["Pelvis"] = ac.calculate_angle_4(
                joints["R_HIP"],
                joints["L_HIP"],
                np.array([0, 0, 0]), np.array([10, 0, 0]))
        
        if self.joint_exist_check(joints, ["L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP"]):
            # 3. 오른쪽 골반 각도 (몸통-골반-무릎"])
            angle_body_twist = ret["Shoulder"] - ret["Pelvis"]
            ret["Body Twist"] = round((angle_body_twist + 180) % 360 - 180, 2)
        
        if self.joint_exist_check(joints, ["R_HIP", "R_KNEE", "R_ANKLE"]):
        # 4. 오른쪽 무릎 각도
            ret["R Knee"] = ac.calculate_angle_3(
                joints["R_HIP"],
                joints["R_KNEE"],
                joints["R_ANKLE"])
        
        if self.joint_exist_check(joints, ["L_HIP", "L_KNEE", "L_ANKLE"]):
            ret["L Knee"] = ac.calculate_angle_3(
                joints["L_HIP"],
                joints["L_KNEE"],
                joints["L_ANKLE"])

        return ret
    
    def joint_exist_check(self, data:dict, joint_list):
        for joint in joint_list:
            if joint not in [*data.keys()]:
                return False
        return True
    
    def items(self):
        return ["R Elbow", "L Elbow", "R Shoulder",
                "L Shoulder", "Shoulder", "Pelvis",
                "Body Twist", "R Knee", "L Knee"]