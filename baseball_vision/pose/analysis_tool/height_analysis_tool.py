from .analysis_tool import AnalysisTool
from ..processed_data import ProcessedData

import numpy as np
import pandas as pd

class HeightAnalysisTool(AnalysisTool):

    def __init__(self):
        pass
    def calc(self, data:ProcessedData):

        landmarks_3d_list = data.get_landmarks_3d()
    
        results = [
            self.calc_height(landmarks_3d) if landmarks_3d is not None else {}
            for landmarks_3d in landmarks_3d_list
        ]
                
        df = pd.DataFrame(results)

        df = df.infer_objects(copy=False)
        df_interpolated = df.interpolate(method='linear', axis=0)

        df_final = df_interpolated.ffill().bfill()

        return df_final

    def calc_height(self, landmarks):

        ret = {}

        for name in self.items():
            if name in landmarks:
                ret[name] = round(-landmarks[name][1] * 100, 2)

        return ret
    
    def items(self):
        return ["R_SHOULDER", "R_ELBOW", "R_WRIST",
                "L_SHOULDER", "L_ELBOW", "L_WRIST",
                "R_HIP", "R_KNEE", "R_ANKLE",
                "L_HIP", "L_KNEE", "L_ANKLE"]

def calc_distance(point1, point2):
    return round(np.linalg.norm(point1 - point2) * 100, 4)