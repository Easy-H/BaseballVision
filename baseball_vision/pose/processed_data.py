import pickle
import numpy as np

class ProcessedData:
    def __init__(self):
        pass
    def initialize(self, width_list:list[int], height_list:list[int],
                   fps:int, frame_cnt:int):
        
        self.raw_video_width_list = width_list
        self.raw_video_height_list = height_list
        self.raw_video_fps = fps
        
        self.raw_img_list_list = []
        self.landmarks_3d_list = []
        self.landmarks_2d_list_list = []
        self.visibility_score_list_list = []

        for i in range(len(width_list)):
            self.raw_img_list_list.append([])
            self.landmarks_2d_list_list.append([])
            self.visibility_score_list_list.append([])

    def add_data_at(self, raw_img_list:list[np.array],
                    landmarks_3d:np.array, landmarks_2d_list:list[np.array], visibility_score_list:list[np.array]):

        for i in range(len(raw_img_list)):
            self.raw_img_list_list[i].append(raw_img_list[i])
            self.landmarks_3d_list.append(landmarks_3d)
            self.landmarks_2d_list_list[i].append(landmarks_2d_list[i])
            self.visibility_score_list_list[i].append(visibility_score_list[i])

    def get_frame_cnt(self):
        return len(self.raw_img_list_list[0])

    def get_raw_img_list(self, idx) -> list[np.array]:
        return self.raw_img_list_list[idx]
    
    def get_landmarks_3d(self) -> list[np.array]:
        return self.landmarks_3d_list
    
    def get_landmarks_2d_list(self, idx) -> list[np.array]:
        return self.landmarks_2d_list_list[idx]
    
    def get_visibility_score_list(self, idx) -> list[np.array]:
        return self.visibility_score_list_list[idx]
    
    def save(self, path:str):
        with open (path, "wb") as fw:
            pickle.dump(self, fw)
        pass

    def load(self, path:str):
        with open (path, "rb") as fw:
            data = pickle.load(fw)
            self.raw_video_width_list = data.raw_video_width_list
            self.raw_video_height_list = data.raw_video_height_list
            self.raw_video_fps = data.raw_video_fps
            self.raw_img_list_list = data.raw_img_list_list
            self.landmarks_3d_list = data.landmarks_3d_list
            self.landmarks_2d_list_list = data.landmarks_2d_list_list
            self.visibility_score_list_list = data.visibility_score_list_list
        pass