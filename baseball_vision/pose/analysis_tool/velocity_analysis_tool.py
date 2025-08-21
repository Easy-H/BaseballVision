from .analysis_tool import AnalysisTool
from ..processed_data import ProcessedData

import numpy as np
from utils.data import PandasDataFrame

class VelocityAnalysisTool(AnalysisTool):

    def __init__(self):
        pass
    def calc(self, data:ProcessedData):
        landmarks_3d_list = data.get_landmarks_3d()
    
        results = [{}]

        before_landmarks = landmarks_3d_list[0]

        for i in range(1, len(landmarks_3d_list)):

            next_landmarks = landmarks_3d_list[i]
            
            if before_landmarks is None or next_landmarks is None:
                results.append({})
            else:
                results.append(self.calc_distance(
                    before_landmarks, next_landmarks))
            
            before_landmarks = next_landmarks

        return PandasDataFrame(results)

    def calc_distance(self, pose_before, pose_next):

        ret = {}

        for name in self.items():
            r = self._calc_distance(name, pose_before, pose_next)
            if r is None:
                continue
            ret[name] = r

        return ret
    
    def _calc_distance(self, name, pose_before, pose_next):
        if not self.check_data_exist(pose_before, [name]):
            return None
        if not self.check_data_exist(pose_next, [name]):
            return None
        
        return round(np.linalg.norm(pose_before[name] - pose_next[name]) * 100, 4)
    
    def items(self):
        return ["R_SHOULDER", "L_SHOULDER", 
                "R_ELBOW", "L_ELBOW", 
                "R_WRIST", "L_WRIST",
                "R_HIP", "L_HIP",
                "R_KNEE", "L_KNEE", 
                "R_ANKLE", "L_ANKLE"]