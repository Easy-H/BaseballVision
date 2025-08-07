from .analysis_tool import AnalysisTool
from ..processed_data import ProcessedData

import numpy as np
import pandas as pd

class VelocityAnalysisTool(AnalysisTool):

    def __init__(self):
        pass
    def calc(self, data:ProcessedData):
        landmarks_3d_list = data.get_landmarks_3d()
    
        results = [{}]

        for i in range(len(landmarks_3d_list) - 1):
            current_landmarks = landmarks_3d_list[i]
            next_landmarks = landmarks_3d_list[i + 1]
            
            if current_landmarks is None or next_landmarks is None:
                results.append({})
            else:
                distance_result = self.calc_distance(
                    current_landmarks, next_landmarks)
                results.append(distance_result)
                
        df = pd.DataFrame(results)

        df = df.infer_objects(copy=False)
        df_interpolated = df.interpolate(method='linear', axis=0)

        df_final = df_interpolated.ffill().bfill()

        return df_final

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
        
        return calc_distance(pose_before[name], pose_next[name])
    
    def items(self):
        return ["R_SHOULDER", "R_ELBOW", "R_WRIST",
                "L_SHOULDER", "L_ELBOW", "L_WRIST",
                "R_HIP", "R_KNEE", "R_ANKLE",
                "L_HIP", "L_KNEE", "L_ANKLE"]

def calc_distance(point1, point2):
    return round(np.linalg.norm(point1 - point2) * 100, 4)