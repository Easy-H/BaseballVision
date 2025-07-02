import open3d as o3d # Open3D 임포트
import numpy as np
import config

class InteractivePoseVisualizer:
    def __init__(self, all_frames_3d_landmarks, connections, initial_fps):
        if not all_frames_3d_landmarks:
            raise ValueError("시각화할 3D 랜드마크 데이터가 없습니다.")

        self.all_frames_3d_landmarks = all_frames_3d_landmarks
        self.num_frames = len(all_frames_3d_landmarks)
        self.connections = connections

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

        all_landmarks_flat = np.vstack(self.all_frames_3d_landmarks)
        self.scene_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(all_landmarks_flat))

        self.scene_widget.setup_camera(1.0, self.scene_bbox, [0, 0, -1])

        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])

        self.pcd_name = "landmarks_pcd"
        self.line_set_name = "connections_line_set"

        self.point_cloud_geometry = None
        self.line_set_geometry = None

        self._initialize_geometry()

        self.window.set_on_key(self._on_key_event)

        print("\n--- 3D 인터랙티브 포즈 애니메이션 조작 방법 ---")
        print("  Spacebar: 재생/일시정지")
        print("  'A' 또는 Left Arrow: 이전 프레임")
        print("  'D' 또는 Right Arrow: 다음 프레임")
        print("  'W' 또는 Up Arrow: 재생 속도 증가")
        print("  'S' 또는 Down Arrow: 재생 속도 감소")
        print("  'R': 카메라 뷰 초기화")
        print("  'Q': 시각화 종료")
        print("---------------------------------------------")

        self._update_geometry_for_frame(self.current_frame_idx)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        # SceneWidget의 frame 속성을 설정하면, 내부적으로 씬의 뷰포트와 크기를 자동으로 조절합니다.
        # 따라서 self.scene.set_view_size나 self.scene.set_viewport는 더 이상 필요하지 않습니다.
        self.scene_widget.frame = r
        # self.scene.set_view_size(r.width, r.height) # 제거
        # self.scene.set_viewport(r.x, r.y, r.width, r.height) # 제거

    def _initialize_geometry(self):
        initial_landmarks = self.all_frames_3d_landmarks[0]

        self.point_cloud_geometry = o3d.geometry.PointCloud()
        self.point_cloud_geometry.points = o3d.utility.Vector3dVector(initial_landmarks)
        self.point_cloud_geometry.colors = o3d.utility.Vector3dVector(
            np.array([[1.0, 0.0, 0.0] for _ in range(initial_landmarks.shape[0])])
        )

        lines = []
        line_colors = []
        for c_tuple in self.connections:
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
        if not (0 <= frame_idx < self.num_frames):
            print(f"경고: 프레임 인덱스 {frame_idx}가 범위를 벗어났습니다 (0-{self.num_frames-1}).")
            return

        frame_landmarks_3d = self.all_frames_3d_landmarks[frame_idx]

        self.point_cloud_geometry.points = o3d.utility.Vector3dVector(frame_landmarks_3d)
        # self.point_cloud_geometry.colors = o3d.utility.Vector3dVector(
        #     np.array([[1.0, 0.0, 0.0] for _ in range(frame_landmarks_3d.shape[0])])
        # )

        self.line_set_geometry.points = o3d.utility.Vector3dVector(frame_landmarks_3d)

    def update_frame_delay(self):
        self.current_frame_delay = self.base_frame_delay / max(0.01, self.playback_speed_factor)

    def _on_key_event(self, event):
        if event.type == o3d.visualization.gui.KeyEvent.Type.DOWN:
            key_code = event.key
            if key_code == o3d.visualization.gui.Key.SPACE:
                self._toggle_play_pause()
            elif key_code == o3d.visualization.gui.Key.A or key_code == o3d.visualization.gui.Key.LEFT:
                self._prev_frame()
            elif key_code == o3d.visualization.gui.Key.D or key_code == o3d.visualization.gui.Key.RIGHT:
                self._next_frame()
            elif key_code == o3d.visualization.gui.Key.W or key_code == o3d.visualization.gui.Key.UP:
                self._speed_up()
            elif key_code == o3d.visualization.gui.Key.S or key_code == o3d.visualization.gui.Key.DOWN:
                self._slow_down()
            elif key_code == o3d.visualization.gui.Key.R:
                self._reset_view()
            elif key_code == o3d.visualization.gui.Key.Q:
                self.window.close()
            return o3d.visualization.gui.Widget.EventCallbackResult.HANDLED
        return o3d.visualization.gui.Widget.EventCallbackResult.IGNORED

    def _toggle_play_pause(self):
        self.is_playing = not self.is_playing
        print(f"애니메이션: {'재생 중' if self.is_playing else '일시 정지'}")

    def _next_frame(self):
        self.is_playing = False # 수동 프레임 이동 시 재생 중지
        self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
        self._update_geometry_for_frame(self.current_frame_idx)
        self.app.post_redraw() # 수동 프레임 이동 후 즉시 갱신 요청

    def _prev_frame(self):
        self.is_playing = False # 수동 프레임 이동 시 재생 중지
        self.current_frame_idx = (self.current_frame_idx - 1 + self.num_frames) % self.num_frames
        self._update_geometry_for_frame(self.current_frame_idx)
        self.app.post_redraw() # 수동 프레임 이동 후 즉시 갱신 요청

    def _speed_up(self):
        self.playback_speed_factor *= 1.2
        self.update_frame_delay()
        print(f"재생 속도: {self.playback_speed_factor:.2f}배")

    def _slow_down(self):
        self.playback_speed_factor /= 1.2
        if self.playback_speed_factor < 0.1: self.playback_speed_factor = 0.1
        self.update_frame_delay()
        print(f"재생 속도: {self.playback_speed_factor:.2f}배")

    def _reset_view(self):
        self.scene_widget.setup_camera(1.0, self.scene_bbox, [0, 0, -1])
        self.app.post_redraw() # 카메라 뷰 변경 후 즉시 갱신 요청
        print("카메라 뷰 초기화.")

    def run(self):
        # 이 함수는 post_to_main_thread에 의해 반복적으로 호출됩니다.
        def update_animation_loop():
            current_time = time.time()
            if self.is_playing and (current_time - self.last_frame_time) >= self.current_frame_delay:
                self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
                self._update_geometry_for_frame(self.current_frame_idx)
                self.last_frame_time = current_time
                self.app.post_redraw() # 프레임 업데이트 후 다시 그리도록 요청

            # 애니메이션 루프를 계속 실행하기 위해 이 함수를 다시 스케줄링합니다.
            self.app.post_to_main_thread(self.window, update_animation_loop)

        self.last_frame_time = time.time()
        # 애니메이션 루프의 첫 호출을 스케줄링하여 시작합니다.
        self.app.post_to_main_thread(self.window, update_animation_loop)
        
        self.app.run()
        self.app.quit()
        print("3D 인터랙티브 시각화가 종료되었습니다.")