import baseball_vision.draw_bone as db
import baseball_vision.filter as filter
import baseball_vision.LandmarkKalmanFilter as LandmarkKalmanFilter
import baseball_vision.draw_bone as db
import config
import cv2
import numpy as np
import os
import sys
import mediapipe as mp
import c3d # C3D library import
import open3d as o3d # Open3D 임포트
import time # time 모듈 임포트

class PoseAnalysisProcessor:
    """
    Handles the entire pose analysis pipeline: video processing,
    MediaPipe pose estimation, Kalman filtering, 2D drawing,
    angle calculation, and saving output videos.
    """
    def __init__(self, output_dir):
        """
        Initializes the PoseAnalysisProcessor.

        Args:
            output_dir (str): Directory to save output videos and C3D files.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=config.MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.kalman_filter_processor = LandmarkKalmanFilter.LandmarkKalmanFilter(
            num_landmarks=len(mp.solutions.pose.PoseLandmark),
            process_noise_std=config.PROCESS_NOISE_STD,
            measurement_noise_std=config.MEASUREMENT_NOISE_STD
        )

    def process_video(self, video_path, video_prename, analysis_tool_func):
        """
        Processes a video file to perform pose estimation, filtering, and analysis.

        Args:
            video_path (str): Path to the input video file.
            video_prename (str): Prefix for output video filenames.
            analysis_tool_func (function): A function (e.g., pitcher_tool, batter_tool)
                                           that takes filtered 3D landmarks and returns
                                           a list of angle strings for display.

        Returns:
            tuple: (all_frames_filtered_3d_landmarks (list), fps (float))
                   Returns empty list and 0.0 if processing fails.
        """
        print("MediaPipe Pose를 초기화합니다...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"오류: 비디오 파일 '{video_path}'를 열 수 없습니다. 파일 경로를 확인하세요.")
            return [], 0.0

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4

        combined_output_path = os.path.join(self.output_dir, video_prename + '_combined_output.mp4')
        bone_output_path = os.path.join(self.output_dir, video_prename + '_bone_output.mp4')

        combined_out = cv2.VideoWriter(combined_output_path, fourcc, fps, (frame_width, frame_height))
        bone_out = cv2.VideoWriter(bone_output_path, fourcc, fps, (frame_width, frame_height))

        if not combined_out.isOpened():
            print(f"오류: 출력 비디오 파일 '{combined_output_path}'를 생성할 수 없습니다. 코덱 또는 권한을 확인하세요.")
            cap.release()
            return [], 0.0
        if not bone_out.isOpened():
            print(f"오류: 출력 비디오 파일 '{bone_output_path}'를 생성할 수 없습니다. 코덱 또는 권한을 확인하세요.")
            combined_out.release()
            cap.release()
            return [], 0.0

        all_frames_filtered_3d_landmarks = [] # To store filtered 3D landmarks for C3D export/3D viz
        
        frame_count = 0
        progress_interval = max(1, total_frames // 100) # Interval for progress display

        print("비디오 처리 시작...")
        while cap.isOpened():
            ret, frame = cap.read() # Read frame
            if not ret:
                if total_frames > 0:
                    sys.stdout.write(f"\r처리 중: 100.00% ({total_frames}/{total_frames} 프레임)")
                    sys.stdout.flush()
                break

            frame_count += 1
            # Display progress
            if total_frames > 0 and (frame_count == 1 or frame_count % progress_interval == 0 or frame_count == total_frames):
                progress_percent = (frame_count / total_frames) * 100
                sys.stdout.write(f"\r처리 중: {progress_percent:.2f}% ({frame_count}/{total_frames} 프레임)")
                sys.stdout.flush()

            # MediaPipe Pose processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.pose_detector.process(frame_rgb)
            frame_rgb.flags.writeable = True

            frame_with_pose = frame.copy()

            if results.pose_landmarks:
                # Extract world landmarks (3D coordinates in meters)
                # Note: MediaPipe's Z-coordinate represents depth relative to the camera.
                # It's not an absolute world Z-axis without camera calibration.
                current_raw_3d_landmarks_array = np.array([
                    [lmk.x, lmk.y, lmk.z]
                    for lmk in results.pose_world_landmarks.landmark
                ])

                # Extract landmark visibility scores
                visibility_scores = np.array([lmk.visibility for lmk in results.pose_world_landmarks.landmark])
                
                # Initialize or filter landmarks with Kalman Filter
                if not config.APPLY_KALMAN_FILTER:
                    filtered_landmarks_array = current_raw_3d_landmarks_array
                if not self.kalman_filter_processor.initialized:
                    self.kalman_filter_processor.initialize_state(current_raw_3d_landmarks_array)
                    filtered_landmarks_array = current_raw_3d_landmarks_array # First frame uses raw data
                else:
                    filtered_landmarks_array = self.kalman_filter_processor.filter(
                        current_raw_3d_landmarks_array, 
                        visibility_scores=visibility_scores, 
                        min_visibility_threshold=0.6
                    )
                        
                all_frames_filtered_3d_landmarks.append(filtered_landmarks_array) # Store filtered landmarks

                # Draw 2D pose on the frame using original MediaPipe landmarks for drawing consistency
                # (filtered landmarks are used for angle calculation)
                frame_with_pose = db.draw_pose_on_frame(frame_with_pose, results.pose_landmarks)
                
                # Calculate and display angles using the analysis tool function
                angle_strings = analysis_tool_func(filtered_landmarks_array) 
            else:
                angle_strings = []

            
            final_pose_only_frame = filter.only_bone(frame, frame_with_pose)
            
            for i, (name, value) in enumerate(angle_strings):
                cv2.putText(frame_with_pose, f"{name}: {value}", 
                            (10, frame.shape[0] - (len(angle_strings) - 1 - i) * 20 - 10), # Adjust Y position
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(final_pose_only_frame, f"{name}: {value}", 
                            (10, frame.shape[0] - (len(angle_strings) - 1 - i) * 20 - 10), # Adjust Y position
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                
            cv2.putText(frame_with_pose, str(frame_count), (10, 50), # Adjust Y position
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(final_pose_only_frame, str(frame_count), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            # Write results to video files
            combined_out.write(frame_with_pose)
            bone_out.write(final_pose_only_frame)

        print("\n비디오 객체를 해제합니다...")
        combined_out.release()
        bone_out.release()
        cap.release()
        self.pose_detector.close() # Close MediaPipe pose detector

        return all_frames_filtered_3d_landmarks, fps