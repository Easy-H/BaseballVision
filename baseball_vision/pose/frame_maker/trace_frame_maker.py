from ..processed_data import ProcessedData
from .pose_frame_maker import IPoseFrameMaker
import config

import cv2
import numpy as np
import pandas as pd

class TraceFrameMaker(IPoseFrameMaker):
    def __init__(self):
        self.target_idx = 0

    def set_data(self, data:ProcessedData, df:pd.DataFrame):
        if data is None: return

        self.bv_data = data
        self.raw_img_list = data.get_raw_img_list(self.target_idx)
        self.landmarks_2d_list = data.get_landmarks_2d_list(self.target_idx)
        self.visibility_list = data.get_visibility_score_list(self.target_idx)
        self.df = df
        
    def get_size(self):
        width = self.bv_data.raw_video_width_list[self.target_idx]
        height = self.bv_data.raw_video_height_list[self.target_idx]
        
        return (int(width), int(height))
    
    def get_img_at(self, idx:int):
        if idx >= self.bv_data.get_frame_cnt():
            return None
        
        img = self.raw_img_list[idx].copy()
        
        return draw_landmark(img,
                             idx,
                             self.landmarks_2d_list,
                             self.get_size(),
                             self.visibility_list,
                             self.labels)
        
        
def draw_landmark(img, idx, landmarks_list_list, img_size, visibility_list_list, labels):
    
    before_coordinate = {}

    for key, coord in landmarks_list_list[0].items():
        if key not in labels:
            continue
        if (visibility_list_list[0][key] < 
            config.MODEL_CONFIG["MIN_DRAW_VISIBILITY"]):
            continue
        
        coordinate = (int(coord[0] * img_size[0]), 
                            int(coord[1] * img_size[1]))

        cv2.circle(img, coordinate, 2, (255, 0, 0), -1)

        before_coordinate[key] = coordinate

    for i in range(idx):
        if landmarks_list_list[i + 1] is None:
            continue
        now_coordinate = {}
        for key, coord in landmarks_list_list[i + 1].items():
            if key not in labels:
                continue
            if (visibility_list_list[i + 1][key] < 
                config.MODEL_CONFIG["MIN_DRAW_VISIBILITY"]):
                continue
        
            coordinate = (int(coord[0] * img_size[0]), 
                                int(coord[1] * img_size[1]))
            
            if key in before_coordinate:
                cv2.line(img, before_coordinate[key],
                        coordinate, (255, 0, 0), 2)
            else:
                cv2.circle(img, coordinate, 2, (255, 0, 0), -1)


            now_coordinate[key] = coordinate

        before_coordinate = now_coordinate

    return img