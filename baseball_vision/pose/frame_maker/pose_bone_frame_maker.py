from ..processed_data import ProcessedData
from .pose_frame_maker import IPoseFrameMaker
import config

import cv2
import numpy as np
import pandas as pd

class PoseFrameMaker(IPoseFrameMaker):
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
        
        return draw_pose_on_frame(
                self.raw_img_list[idx],
                self.landmarks_2d_list[idx],
                self.visibility_list[idx])
    
class PoseOnlyFrameMaker(IPoseFrameMaker):
    def __init__(self, data:ProcessedData = None, df:pd.DataFrame = None):
        self.set_data(data, df)
        self.target_idx = 0

    def set_data(self, data:ProcessedData, df:pd.DataFrame):
        if data is None: return

        self.bv_data = data
        self.landmark_2d_list = data.get_landmarks_2d_list(self.target_idx)
        self.visibility_list = data.get_visibility_score_list(self.target_idx)
        self.df = df
        
    def get_size(self):
        width = self.bv_data.raw_video_width_list[self.target_idx]
        height = self.bv_data.raw_video_height_list[self.target_idx]
        
        return (int(width), int(height))
    
    def get_img_at(self, idx:int):
        if idx >= self.bv_data.get_frame_cnt():
            return None
        
        img = np.zeros_like(
            self.bv_data.get_raw_img_list(self.target_idx)[0])

        return draw_pose_on_frame(
                img,
                self.landmark_2d_list[idx],
                self.visibility_list[idx])
    
def draw_landmarks_custom(image, landmarks_array, image_width, image_height, visibility_array):
    
    for key, coord in landmarks_array.items():
        landmark_x = coord[0]
        landmark_y = coord[1]
        landmark_visibility = visibility_array[key]
        
        if landmark_visibility < config.MODEL_CONFIG["MIN_DRAW_VISIBILITY"]:
            continue
        
        center_coordinates = (int(landmark_x * image_width), int(landmark_y * image_height))

        # Accessing MediaPipe's PoseLandmark enum values for comparison
        if key == 'NOSE': # Nose: single face node
            cv2.circle(image, center_coordinates, 5, (255, 255, 255), -1) # White circle
        elif key not in config.JOINTS:
            pass # Don't draw these facial landmarks'''
        else: # Body, arm, leg landmarks: small gray dot
            cv2.circle(image, center_coordinates, 2, (64, 64, 64), -1)
        # Facial landmarks (eyes, ears, mouth) are typically from 1 to 10

def draw_connections_custom(image, landmarks_array, image_width, image_height, visibility_array):

    for connection in config.POSE_CONNECTIONS:
        idx1, idx2 = connection

        if visibility_array[idx1] < config.MODEL_CONFIG["MIN_DRAW_VISIBILITY"] \
           or visibility_array[idx2] < config.MODEL_CONFIG["MIN_DRAW_VISIBILITY"]:
            continue
        
        # Get color for the connection from the predefined map
        color = config.CONNECTIONS_COLORS.get(connection, None)
        if color is None: # Check if tuple order is reversed in map (for robustness)
            color = config.CONNECTIONS_COLORS.get((idx2, idx1), None)

        if color is not None:
            point1 = (int(landmarks_array[idx1][0] * image_width),
                      int(landmarks_array[idx1][1] * image_height))
            point2 = (int(landmarks_array[idx2][0] * image_width),
                      int(landmarks_array[idx2][1] * image_height))
            cv2.line(image, point1, point2, color, 2) # Line thickness 2

def draw_pose_on_frame(background_img, pose_landmarks_2d, visibility_score=None):

    if pose_landmarks_2d is None:
        return background_img
    if visibility_score is None:
        return background_img
    
    img = background_img.copy()
    h, w, _ = background_img.shape

    # Call functions to draw landmarks and connections
    draw_connections_custom(img, pose_landmarks_2d, w, h, visibility_score)
    draw_landmarks_custom(img, pose_landmarks_2d, w, h, visibility_score)
    
    return img