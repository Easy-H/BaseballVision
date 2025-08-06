import open3d as o3d
import numpy as np
import config
import time # 현재 시간을 위해 time 임포트

import os # 경로 조작을 위해 os 임포트

class InteractivePoseVisualizer:
    def __init__(self, landmarks_data_dir, total_frames, initial_fps):
        if not os.path.isdir(landmarks_data_dir) or total_frames == 0:
            raise ValueError("시각화할 3D 랜드마크 데이터가 없습니다.")

        self.landmarks_data_dir = landmarks_data_dir
        self.total_frames = total_frames
        
        self.current_frame_idx = 0
        self.is_playing = False
        self.playback_speed_factor = 1.0
        self.base_frame_delay = 1.0 / initial_fps if initial_fps > 0 else 0.033
        self.update_frame_delay()

        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window("3D Interactive Pose Animation", 1024, 768)
        self.scene_widget = o3d.visualization.gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)
        self.window.set_on_layout(self._on_layout)

        # 바운딩 박스 및 초기 형상 설정을 위한 초기 랜드마크 가져오기
        initial_landmarks = self._load_frame_landmarks(0)
        if initial_landmarks is None:
            raise ValueError("초기 랜드마크 데이터를 로드할 수 없습니다.")

        # 장면 BBox를 계산하려면 프레임 샘플이 필요하거나 합리적인 크기를 가정해야 합니다.
        # 더 강력한 방법은 시작 시간이 허용하는 경우 모든 랜드마크 파일을 한 번 반복하거나,
        # 이동이 제한된 경우 몇 개의 초기 및 최종 프레임을 기반으로 계산하는 것입니다.
        # 현재는 initial_landmarks만으로 초기 카메라를 설정하기에 충분하다고 가정합니다.
        # all_frames_3d_landmarks가 없는 경우 BBox에 대한 더 나은 접근 방식은 더 복잡하지만
        # 처리 중 미리 계산/저장하거나 로드하여 수행할 수 있습니다.
        # 현재는 임시 바운딩 박스를 만듭니다.
        # 임시 해결책은 다음과 같습니다.
        min_coords = np.min(initial_landmarks, axis=0) - 0.5 # 약간의 패딩 추가
        max_coords = np.max(initial_landmarks, axis=0) + 0.5
        self.scene_bbox = o3d.geometry.AxisAlignedBoundingBox(min_coords, max_coords)

        self.scene_widget.setup_camera(1.0, self.scene_bbox, [0, 0, -1])
        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])

        self.pcd_name = "landmarks_pcd"
        self.line_set_name = "connections_line_set"

        self.point_cloud_geometry = None
        self.line_set_geometry = None

        self._initialize_geometry(initial_landmarks)
        self.window.set_on_key(self._on_key_event)

        print("\n--- 3D 인터랙티브 포즈 애니메이션 조작 방법 ---")
        print(f"  총 프레임: {self.total_frames}")
        print("  Spacebar: 재생/일시정지")
        print("  'A' 또는 Left Arrow: 이전 프레임")
        print("  'D' 또는 Right Arrow: 다음 프레임")
        print("  'W' 또는 Up Arrow: 재생 속도 증가")
        print("  'S' 또는 Down Arrow: 재생 속도 감소")
        print("  'R': 카메라 뷰 초기화")
        print("  'Q': 시각화 종료")
        print("---------------------------------------------")

        self._update_geometry_for_frame(self.current_frame_idx)

    def _load_frame_landmarks(self, frame_idx):
        file_path = os.path.join(self.landmarks_data_dir, f"frame_{frame_idx:05d}.npy")
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            print(f"경고: 프레임 파일 없음: {file_path}")
            return None

    def _initialize_geometry(self, initial_landmarks):
        self.point_cloud_geometry = o3d.geometry.PointCloud()
        self.point_cloud_geometry.points = o3d.utility.Vector3dVector(initial_landmarks)
        self.point_cloud_geometry.colors = o3d.utility.Vector3dVector(
            np.array([[1.0, 0.0, 0.0] for _ in range(initial_landmarks.shape[0])])
        )

        lines = []
        line_colors = []
        for c_tuple in config.mp_pose.POSE_CONNECTIONS:
            lines.append([c_tuple[0], c_tuple[1]])
            mapped_color = None
            for conn_key, color_rgb in config.CONNECTIONS_COLORS.items():
                if (conn_key[0] == c_tuple[0] and conn_key[1] == c_tuple[1]) or \
                   (conn_key[0] == c_tuple[1] and conn_key[1] == c_tuple[0]):
                    mapped_color = [x / 255.0 for x in color_rgb]
                    break
            line_colors.append(mapped_color if mapped_color else [1.0, 1.0, 1.0])

        self.line_set_geometry = o3d.geometry.LineSet()
        self.line_set_geometry.points = o3d.utility.Vector3dVector(initial_landmarks)
        self.line_set_geometry.lines = o3d.utility.Vector2iVector(lines)
        self.line_set_geometry.colors = o3d.utility.Vector3dVector(np.array(line_colors))

        red_material = o3d.visualization.rendering.MaterialRecord()
        red_material.base_color = [1.0, 0.0, 0.0, 1.0]
        red_material.shader = "defaultLit"

        white_material = o3d.visualization.rendering.MaterialRecord()
        white_material.base_color = [1.0, 1.0, 1.0, 1.0]
        white_material.shader = "defaultLit"

        self.scene_widget.scene.add_geometry(self.pcd_name, self.point_cloud_geometry, red_material)
        self.scene_widget.scene.add_geometry(self.line_set_name, self.line_set_geometry, white_material)


    def _update_geometry_for_frame(self, frame_idx):
        if not (0 <= frame_idx < self.total_frames):
            print(f"경고: 프레임 인덱스 {frame_idx}가 범위를 벗어났습니다 (0-{self.total_frames-1}).")
            return

        frame_landmarks_3d = self._load_frame_landmarks(frame_idx)
        if frame_landmarks_3d is None:
            # 프레임이 누락된 경우, 원하는 동작에 따라 이전 프레임의 포즈를 유지하거나
            # 장면을 지울 수 있습니다.
            print(f"경고: 프레임 {frame_idx}의 랜드마크 데이터를 로드할 수 없습니다. 이전 프레임을 유지합니다.")
            return

        self.point_cloud_geometry.points = o3d.utility.Vector3dVector(frame_landmarks_3d)
        self.line_set_geometry.points = o3d.utility.Vector3dVector(frame_landmarks_3d)

        self.scene_widget.scene.update_geometry(self.pcd_name, self.point_cloud_geometry)
        self.scene_widget.scene.update_geometry(self.line_set_name, self.line_set_geometry)
        
    # ... (나머지 메서드: _on_layout, update_frame_delay, _on_key_event,
    # _toggle_play_pause, _next_frame, _prev_frame, _speed_up, _slow_down, _reset_view) ...

    def _next_frame(self):
        self.is_playing = False # 수동 탐색 시 재생 중지
        self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
        self._update_geometry_for_frame(self.current_frame_idx)
        self.app.post_redraw()

    def _prev_frame(self):
        self.is_playing = False # 수동 탐색 시 재생 중지
        self.current_frame_idx = (self.current_frame_idx - 1 + self.total_frames) % self.total_frames
        self._update_geometry_for_frame(self.current_frame_idx)
        self.app.post_redraw()

    def run(self):
        def update_animation_loop():
            current_time = time.time()
            if self.is_playing and (current_time - self.last_frame_time) >= self.current_frame_delay:
                self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
                self._update_geometry_for_frame(self.current_frame_idx)
                self.last_frame_time = current_time
                self.app.post_redraw()

            if self.window.is_closed(): # 창이 닫혔는지 확인하여 스케줄링 중지
                return

            self.app.post_to_main_thread(self.window, update_animation_loop)

        self.last_frame_time = time.time()
        self.app.post_to_main_thread(self.window, update_animation_loop)

        self.app.run()
        self.app.quit()
        print("3D 인터랙티브 시각화가 종료되었습니다.")