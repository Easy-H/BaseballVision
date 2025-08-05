import numpy as np
from . import angle_calc as ac
from .pitcher_tool import PitcherAnalysisTool

class BatterAnalysisTool(PitcherAnalysisTool):
    # [수정 사항 17] __init__ 메서드: 상위 클래스의 초기화를 호출하고, focus_label 기본값 설정
    def __init__(self, label:list=[]): 
        super().__init__(label) # PitcherAnalysisTool의 __init__ 호출
        if not label: # 전달된 label이 없으면 기본 focus_label 설정
            self.focus_label = ["R Elbow", "R Shoulder", "Shoulder", "Pelvis", "Body Twist", "R Knee"] # 타자 중점 각도

    def calc_joints(self, joints):
        result = super().calc_joints(joints)
        # 손목 각도 추가
        
        if self.joint_exist_check(joints, ["R_ELBOW", "R_WRIST", "R_PINKY_TIP"]):
            result["R Wrist"] = ac.calculate_angle_3(
                joints["R_ELBOW"],
                joints["R_WRIST"],
                joints["R_PINKY_TIP"])
        
        if self.joint_exist_check(joints, ["L_ELBOW", "L_WRIST", "L_PINKY_TIP"]):
            result["L Wrist"] = ac.calculate_angle_3(
                joints["L_ELBOW"],
                joints["L_WRIST"],
                joints["L_PINKY_TIP"])
        
        return result
    
    def items(self):
        ret = super.items()
        ret.append("R Wrist")
        ret.append("L Wrist")
        return ret