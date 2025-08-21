from .analysis_tool import AnalysisTool
from ..processed_data import ProcessedData

import numpy as np
from utils.data import PandasDataFrame

class HeightAnalysisTool(AnalysisTool):

    def __init__(self):
        pass
    def calc(self, data:ProcessedData):

        landmarks_3d_list = data.get_landmarks_3d()
    
        results = [
            self.calc_height(landmarks_3d) if landmarks_3d is not None else {}
            for landmarks_3d in landmarks_3d_list
        ]
        
        return PandasDataFrame(results)

    def calc_height(self, landmarks):

        ret = {}

        for name in self.items():
            if name in landmarks:
                ret[name] = round(landmarks[name][1] * 100, 2)

        return ret
    
    def items(self):
        return ["R_SHOULDER", "L_SHOULDER", 
                "R_ELBOW", "L_ELBOW", 
                "R_WRIST", "L_WRIST",
                "R_HIP", "L_HIP",
                "R_KNEE", "L_KNEE", 
                "R_ANKLE", "L_ANKLE"]