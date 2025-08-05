import cv2
from .frame_maker import IFrameMaker

class VideoMaker:
    def __init__(self):
        pass

    def save_video(self, path:str, frame_maker:IFrameMaker, fps:int = 24):

        ret_video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps, frame_maker.get_size())

        if not ret_video.isOpened():
            print(f"오류: 출력 비디오 파일 '{path}'를 생성할 수 없습니다. 코덱 또는 권한을 확인하세요.")
            return
        
        i = 0
        while True:
            new_frame = frame_maker.get_img_at(i)
            if new_frame is None:
                break
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
            ret_video.write(new_frame)
            i += 1
        
        ret_video.release()