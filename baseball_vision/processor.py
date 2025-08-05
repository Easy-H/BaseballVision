from .pose_detector import IPoseDetector, MediaPipePoseDetector
from baseball_vision.processed_data import ProcessedData
from utils.progress_bar.progress_bar import IProgressBar
import cv2

class Processor:
    def __init__(self):
        self.pose_detector = None
        self.output_dir = None
        self.landmarks_data_dir = None
        self.baseball_vision_data = None

    def setting(self,
                pose_detector:IPoseDetector=MediaPipePoseDetector(),
                progress:IProgressBar=IProgressBar()):
        self.pose_detector = pose_detector
        self.progress_bar = progress

    def process_video(self, video_path_list:list):
        
        print("MediaPipe Pose를 초기화합니다...")
        
        ret = self._initialize_data(video_path_list)
        
        if ret is None:
            return None

        if self.caps is None: # Check if initialization failed
            return None

        self._process_video(ret)
        self.pose_detector.close()

        return ret
    
    def _process_video(self, data:ProcessedData):

        img_list_list = []

        while True:
            img_list = self.get_img_list()
            if img_list is None:
                break
            img_list_list.append(img_list)
            
        print("비디오 처리 시작...")
        frame_count = 0
        for img_list in img_list_list:
            img_list = self._img_list_bgr_to_rgb(img_list)
            # Process the current frame for pose estimation and angle calculation
            landmarks_3d, landmarks_2d_list, visibility_score_list = \
                self._process_frame_for_pose(img_list)
            data.add_data_at(img_list, landmarks_3d,
                             landmarks_2d_list, visibility_score_list)
        
            frame_count += 1
            self.progress_bar.display_progress(frame_count, len(img_list_list) - 1)

            
        print("\n비디오 객체를 해제합니다...")

        for cap in self.caps:
            cap.release()
            
    def get_img_list(self):
        img_list = []
        for cap in self.caps:
            ret, img = cap.read() # Read frame
            if not ret:
                return None
            img_list.append(img)
        return img_list
    
    def _initialize_data(self, video_path_list:list[str]):
        
        self.caps = []
        
        img_width_list = []
        img_height_list = []

        for path in video_path_list:
            cap = cv2.VideoCapture(path)

            if not cap.isOpened():
                print(f"오류: 비디오 파일 '{path}'를 열 수 없습니다. 파일 경로를 확인하세요.")
                return None
            
            img_width_list.append(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height_list.append(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.caps.append(cap)

        fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        total_frame_cnt = int(self.caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

        ret = ProcessedData()
        ret.initialize(img_width_list, img_height_list,
                       fps, total_frame_cnt)
        return ret
    
    def _img_list_bgr_to_rgb(self, img_list):

        img_rgb_list = []

        for img in img_list:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb_list.append(img_rgb)

        return img_rgb_list

    def _process_frame_for_pose(self, img_list):
        
        for img_rgb in img_list:
            img_rgb.flags.writeable = False

        landmarks_3d, landmarks_2d_list, visibility_score_list = \
            self.pose_detector.process(img_list)
        
        for img_rgb in img_list:
            img_rgb.flags.writeable = True

        return landmarks_3d, landmarks_2d_list, visibility_score_list