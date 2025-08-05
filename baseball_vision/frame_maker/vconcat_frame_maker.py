from ..processed_data import ProcessedData
from utils.frame_maker import IFrameMaker
import config

import cv2
import numpy as np
import pandas as pd

class VConcatFrameMaker(IFrameMaker):
    def __init__(self, frame_maker_list:IFrameMaker):
        self.frame_maker_list = frame_maker_list
        
    def get_size(self):
        width = 0
        height = 0

        for frame_maker in self.frame_maker_list:
            (w, h) = frame_maker.get_size()
            width = w
            height += h
        
        return (int(width), int(height))
    
    def get_img_at(self, idx:int):

        img_list = []

        for frame_maker in self.frame_maker_list:
            img = frame_maker.get_img_at(idx)
            if img is None:
                return None
            img_list.append(img)

        ret_img = cv2.vconcat(img_list)

        return ret_img

    def overlay_df_data(self, img, data, frame_cnt):
        ret_img = img.copy()

        y_offset = ret_img.shape[0] - (len(data) * 20) # Start from bottom, reserving space for all lines
        for i, (name, value) in enumerate(data.items()):
            cv2.putText(ret_img, f"{name}: {str(value)}",
                        (10, y_offset + i * 20 - 10), # Adjust Y position for each line
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add frame count
            cv2.putText(ret_img, str(frame_cnt), (10, 30), # Move frame count to top-left
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Vertically concatenate the cur
        return ret_img