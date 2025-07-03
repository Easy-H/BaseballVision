import baseball_vision.draw_image as di
from baseball_vision.LandmarkKalmanFilter import LandmarkKalmanFilter
import config
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys

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
        self.kalman_filter_processor = LandmarkKalmanFilter(
            num_landmarks=len(mp.solutions.pose.PoseLandmark),
            process_noise_std=config.PROCESS_NOISE_STD,
            measurement_noise_std=config.MEASUREMENT_NOISE_STD
        )
        # 클래스 인스턴스에 비디오 프레임 너비/높이 저장
        self.frame_width = 0
        self.frame_height = 0

    def _initialize_video_capture_and_writers(self, video_path, video_prename, graph_height):
        """
        Initializes video capture and output video writers.

        Args:
            video_path (str): Path to the input video file.
            video_prename (str): Prefix for output video filenames.
            graph_height (int): Height in pixels to allocate for the graph area below the video.

        Returns:
            tuple: (cap, combined_out, bone_out, fps, total_frames)
                   Returns None for all if initialization fails.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"오류: 비디오 파일 '{video_path}'를 열 수 없습니다. 파일 경로를 확인하세요.")
            return None, None, None, 0.0, 0

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4

        combined_output_path = os.path.join(self.output_dir, video_prename + '_combined_output.mp4')
        bone_output_path = os.path.join(self.output_dir, video_prename + '_bone_output.mp4')

        # 출력 비디오의 높이는 원본 프레임 높이 + 그래프 높이
        output_frame_size = (self.frame_width, self.frame_height + graph_height)

        combined_out = cv2.VideoWriter(combined_output_path, fourcc, fps, output_frame_size)
        bone_out = cv2.VideoWriter(bone_output_path, fourcc, fps, output_frame_size)

        if not combined_out.isOpened():
            print(f"오류: 출력 비디오 파일 '{combined_output_path}'를 생성할 수 없습니다. 코덱 또는 권한을 확인하세요.")
            cap.release()
            return None, None, None, 0.0, 0
        if not bone_out.isOpened():
            print(f"오류: 출력 비디오 파일 '{bone_output_path}'를 생성할 수 없습니다. 코덱 또는 권한을 확인하세요.")
            combined_out.release()
            cap.release()
            return None, None, None, 0.0, 0

        return cap, combined_out, bone_out, fps, total_frames

    def _process_frame_for_pose(self, frame, analysis_tool):
        """
        Processes a single frame: performs MediaPipe pose estimation, Kalman filtering,
        and calculates angles.

        Args:
            frame (np.array): The current video frame (BGR).
            analysis_tool (object): An object with 'calc' and 'skip' methods for angle calculation.

        Returns:
            tuple: (frame_with_pose, final_pose_only_frame, filtered_landmarks_array, tool_output)
                   Returns (frame_copy, frame_copy_diff, None, {}) if no pose landmarks are detected.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        frame_with_pose = frame.copy()
        filtered_landmarks_array = None
        tool_output = {}

        if results.pose_landmarks:
            current_raw_3d_landmarks_array = np.array([
                [lmk.x, lmk.y, lmk.z]
                for lmk in results.pose_world_landmarks.landmark
            ])
            visibility_scores = np.array([lmk.visibility for lmk in results.pose_world_landmarks.landmark])

            if not config.APPLY_KALMAN_FILTER:
                filtered_landmarks_array = current_raw_3d_landmarks_array
            elif not self.kalman_filter_processor.initialized:
                self.kalman_filter_processor.initialize_state(current_raw_3d_landmarks_array)
                filtered_landmarks_array = current_raw_3d_landmarks_array # First frame uses raw data
            else:
                filtered_landmarks_array = self.kalman_filter_processor.filter(
                    current_raw_3d_landmarks_array,
                    visibility_scores=visibility_scores,
                    min_visibility_threshold=0.6
                )

            frame_with_pose = di.draw_pose_on_frame(frame_with_pose, results.pose_landmarks)
            tool_output = analysis_tool.calc(filtered_landmarks_array)
        else:
            analysis_tool.skip() # Signal to analysis tool that no landmarks were found

        final_pose_only_frame = di.draw_diff(frame, frame_with_pose)

        return frame_with_pose, final_pose_only_frame, filtered_landmarks_array, tool_output

    def _add_overlays_and_graph(self, original_frames_with_pose, tool_output, frame_count, total_frames, graph_height, analysis_tool):
        """
        Adds angle text, frame count, and a graph image to each given frame.

        Args:
            original_frames_with_pose (list): A list of frames (e.g., [frame_with_pose, final_pose_only_frame]).
            tool_output (dict): Dictionary of angle names and values.
            frame_count (int): Current frame number.
            total_frames (int): Total number of frames.
            graph_height (int): Height allocated for the graph.
            analysis_tool (object): An object with a 'create_graph_image' method.

        Returns:
            list: A list of processed frames with overlays and concatenated graph.
        """
        # Create and add graph image
        graph_img = analysis_tool.create_graph_image(
            frame_count,
            total_frames,
            labels=tool_output.keys(),
            width=self.frame_width,
            height=graph_height
        )

        # Ensure graph image matches frame width
        if graph_img.shape[1] != self.frame_width:
            graph_img = cv2.resize(graph_img, (self.frame_width, graph_img.shape[0]))
        
        processed_frames = []
        
        for frame_item in original_frames_with_pose:
            # Create a copy to avoid modifying the original frame in place for each loop
            current_frame = frame_item.copy() 

            # Add angle text
            y_offset = current_frame.shape[0] - (len(tool_output) * 20) # Start from bottom, reserving space for all lines
            for i, (name, value) in enumerate(tool_output.items()):
                cv2.putText(current_frame, f"{name}: {str(value)}",
                            (10, y_offset + i * 20 - 10), # Adjust Y position for each line
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add frame count
            cv2.putText(current_frame, str(frame_count), (10, 30), # Move frame count to top-left
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            # Vertically concatenate the current frame with the graph
            combined_frame = cv2.vconcat([current_frame, graph_img])
            processed_frames.append(combined_frame)

        return processed_frames


    def process_video(self, video_path, video_prename, analysis_tool, graph_height=200):
        """
        Processes a video file to perform pose estimation, filtering, and analysis.

        Args:
            video_path (str): Path to the input video file.
            video_prename (str): Prefix for output video filenames.
            analysis_tool (object): An object with 'calc', 'skip', and 'create_graph_image' methods.
            graph_height (int): Height in pixels to allocate for the graph area below the video (default: 200).

        Returns:
            tuple: (all_frames_filtered_3d_landmarks (list), fps (float))
                   Returns empty list and 0.0 if processing fails.
        """
        print("MediaPipe Pose를 초기화합니다...")

        cap, combined_out, bone_out, fps, total_frames = \
            self._initialize_video_capture_and_writers(video_path, video_prename, graph_height)

        if cap is None: # Check if initialization failed
            return [], 0.0

        all_frames_filtered_3d_landmarks = [] # To store filtered 3D landmarks for C3D export/3D viz

        frame_count = 0
        progress_interval = max(1, total_frames // 100) # Interval for progress display

        print("비디오 처리 시작...")
        while cap.isOpened():
            ret, frame = cap.read() # Read frame
            if not ret:
                # Ensure 100% progress is displayed at the end
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

            # Process the current frame for pose estimation and angle calculation
            frame_with_pose, final_pose_only_frame, filtered_landmarks_array, tool_output = \
                self._process_frame_for_pose(frame, analysis_tool)

            if filtered_landmarks_array is not None:
                all_frames_filtered_3d_landmarks.append(filtered_landmarks_array)

            # Add text, frame count, and graph to both frames
            processed_frames = self._add_overlays_and_graph(
                [frame_with_pose, final_pose_only_frame], tool_output, frame_count,
                total_frames, graph_height, analysis_tool
            )
            
            # Write results to video files
            combined_out.write(processed_frames[0]) # frame_with_pose + graph
            bone_out.write(processed_frames[1])     # final_pose_only_frame + graph

        print("\n비디오 객체를 해제합니다...")
        combined_out.release()
        bone_out.release()
        cap.release()
        analysis_tool.run() # Finalization step for the analysis_tool, e.g., saving data
        self.pose_detector.close() # Close MediaPipe pose detector

        return all_frames_filtered_3d_landmarks, fps