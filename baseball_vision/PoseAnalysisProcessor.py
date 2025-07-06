import baseball_vision.draw_image as di
import baseball_vision.PoseDetector as bvpd
import config
import numpy as np
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
        self.pose_detector = bvpd.MediaPipePoseDetector()
        
        # 클래스 인스턴스에 비디오 프레임 너비/높이 저장
        self.frame_width = 0
        self.frame_height = 0
        self.frame_count = 0

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
        caps = []
        
        for i in range(len(video_path)):
            cap = cv2.VideoCapture(video_path[i])
            if not cap.isOpened():
                print(f"오류: 비디오 파일 '{video_path[i]}'를 열 수 없습니다. 파일 경로를 확인하세요.")
                return None, None, None, 0
            caps.append(cap)

        self.frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        self.total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4

        self.combined_output_path = os.path.join(self.output_dir, video_prename + '_combined_output.mp4')
        bone_output_path = os.path.join(self.output_dir, video_prename + '_bone_output.mp4')

        # 출력 비디오의 높이는 원본 프레임 높이 + 그래프 높이
        output_frame_size = (self.frame_width, self.frame_height + graph_height)

        combined_out = cv2.VideoWriter(self.combined_output_path, fourcc, fps, output_frame_size)
        bone_out = cv2.VideoWriter(bone_output_path, fourcc, fps, output_frame_size)

        if not combined_out.isOpened():
            print(f"오류: 출력 비디오 파일 '{self.combined_output_path}'를 생성할 수 없습니다. 코덱 또는 권한을 확인하세요.")
            for i in range(len(caps)):
                caps[i].release()
            return None, None, None, 0
        if not bone_out.isOpened():
            print(f"오류: 출력 비디오 파일 '{bone_output_path}'를 생성할 수 없습니다. 코덱 또는 권한을 확인하세요.")
            for i in range(len(caps)):
                caps[i].release()
            return None, None, None, 0

        return caps, combined_out, bone_out

    def _add_overlays_and_graph(self, original_frames, tool_output, graph_height, analysis_tool):
        """
        Adds angle text, frame count, and a graph image to each given frame.

        Args:
            original_frames (list): A list of frames (e.g., [frame_with_pose, final_pose_only_frame]).
            tool_output (dict): Dictionary of angle names and values.
            graph_height (int): Height allocated for the graph.
            analysis_tool (object): An object with a 'create_graph_image' method.

        Returns:
            list: A list of processed frames with overlays and concatenated graph.
        """
        # Create and add graph image
        graph_img = analysis_tool.create_graph_image(
            self.frame_count,
            self.total_frames,
            width=self.frame_width,
            height=graph_height
        )
        
        processed_frames = []
        
        for frame_item in original_frames:
            # Create a copy to avoid modifying the original frame in place for each loop
            current_frame = frame_item.copy() 

            # Add angle text
            y_offset = current_frame.shape[0] - (len(tool_output) * 20) # Start from bottom, reserving space for all lines
            for i, (name, value) in enumerate(tool_output.items()):
                cv2.putText(current_frame, f"{name}: {str(value)}",
                            (10, y_offset + i * 20 - 10), # Adjust Y position for each line
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add frame count
            cv2.putText(current_frame, str(self.frame_count), (10, 30), # Move frame count to top-left
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            # Vertically concatenate the current frame with the graph
            combined_frame = cv2.vconcat([current_frame, graph_img])
            processed_frames.append(combined_frame)

        return processed_frames
        
    def _process_frame_for_pose(self, frame, analysis_tool, graph_height):
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
        frame_rgbs = []
        for i in range(len(frame)):
            frame_rgb = cv2.cvtColor(frame[0], cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            frame_rgbs.append(frame_rgb)
            
        landmarks_3d, landmarks, visibility_scores = self.pose_detector.process(frame_rgbs)
        
        for i in range(len(frame)):
            frame_rgbs[i].flags.writeable = True

        tool_output = analysis_tool.calc(landmarks_3d)

        frame_with_pose = frame[0].copy()
        frame_with_pose = di.draw_pose_on_frame(frame[0], landmarks[0],
                                                pose_visibility_array=visibility_scores[0])
        final_pose_only_frame = di.draw_diff(frame[0], frame_with_pose)
        
        # Add text, frame count, and graph to both frames
        processed_frames = self._add_overlays_and_graph(
            [frame_with_pose, final_pose_only_frame], tool_output, graph_height, analysis_tool
        )

        return processed_frames

    def process_video(self, video_path:list, video_prename, analysis_tool, graph_height=200):
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

        caps, combined_out, bone_out = \
            self._initialize_video_capture_and_writers(video_path, video_prename, graph_height)

        if caps is None: # Check if initialization failed
            return [], 0.0

        self.frame_count = 0
        progress_interval = max(1, self.frame_count // 100) # Interval for progress display

        print("비디오 처리 시작...")
        while caps[0].isOpened():
            ret = []
            frame = []
            for i in range(len(caps)):
                r, f = caps[i].read() # Read frame
                ret.append(r)
                frame.append(f)
            if not ret[0]:
                # Ensure 100% progress is displayed at the end
                if self.frame_count > 0:
                    sys.stdout.write(f"\r처리 중: 100.00% ({self.frame_count}/{self.frame_count} 프레임)\n")
                    sys.stdout.flush()
                break

            self.frame_count += 1
            # Display progress
            if self.frame_count > 0 and (self.frame_count == 1 or self.frame_count % progress_interval == 0\
                                     or self.frame_count == self.total_frames):
                progress_percent = (self.frame_count / self.total_frames) * 100
                sys.stdout.write(f"\r처리 중: {progress_percent:.2f}% ({self.frame_count}/{self.total_frames} 프레임)")
                sys.stdout.flush()

            # Process the current frame for pose estimation and angle calculation
            processed_frames = self._process_frame_for_pose(frame, analysis_tool, graph_height)
            
            # Write results to video files
            combined_out.write(processed_frames[0]) # frame_with_pose + graph
            bone_out.write(processed_frames[1])     # final_pose_only_frame + graph

        print("\n비디오 객체를 해제합니다...")
        combined_out.release()
        bone_out.release()
        for i in range(len(caps)):
            caps[i].release()
        analysis_tool.run() # Finalization step for the analysis_tool, e.g., saving data
        self.pose_detector.close() # Close MediaPipe pose detector

        return self.combined_output_path, self.pose_detector.get_all_frames_landmarks_3d(), self.total_frames