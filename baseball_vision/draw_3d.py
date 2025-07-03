import open3d as o3d # Open3D 임포트
import config
import time
import numpy as np
import c3d # C3D library import

def show_3d_video(all_frames_3d_landmarks, fps):
    """
    수집된 3D 랜드마크 데이터를 Open3D를 사용하여 애니메이션으로 시각화합니다.

    Args:
        all_frames_3d_landmarks (list): 각 프레임의 3D 랜드마크 (Numpy 배열 리스트).
                                        각 배열은 (num_markers, 3) 형태. (미터 단위)
        connections (list): 랜드마크 연결 정보를 담은 튜플 리스트 (예: [(0, 1), (1, 2)]).
                            MediaPipe의 PoseLandmark.value 인덱스를 사용.
        fps (float): 시각화 속도를 조절하기 위한 초당 프레임 수.
    """
    if not all_frames_3d_landmarks:
        print("시각화할 3D 랜드마크 데이터가 없습니다.")
        return

    # Open3D는 기본적으로 밀리미터 단위를 선호하지만, 여기서는 MediaPipe의 미터 단위를 그대로 사용합니다.
    # 필요하다면 여기서 * 1000.0 하여 밀리미터로 변환할 수 있습니다.
    # 예를 들어: initial_landmarks = all_frames_3d_landmarks[0] * 1000.0

    # 첫 프레임의 랜드마크로 초기 포인트 클라우드 및 라인셋 생성
    initial_landmarks = all_frames_3d_landmarks[0] # (num_markers, 3) 형태

    # 포인트 클라우드 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(initial_landmarks)
    # 모든 랜드마크를 빨간색으로 표시
    pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0] for _ in range(initial_landmarks.shape[0])]))

    # 라인셋 객체 생성 (Open3D는 라인을 나타내기 위해 [point_idx1, point_idx2] 형태의 리스트를 원함)
    lines = []
    line_colors = [] # 각 연결선에 대한 색상 리스트
    for c_tuple in config.mp_pose.POSE_CONNECTIONS:
        lines.append([c_tuple[0], c_tuple[1]]) # 연결선의 점 인덱스 추가
        
        # 미리 정의된 CONNECTIONS_COLORS 딕셔너리에서 색상을 찾아 적용
        mapped_color = None
        for conn_key, color_rgb in config.CONNECTIONS_COLORS.items():
            # 양방향 연결을 모두 고려 (예: (11, 13) 또는 (13, 11))
            if (conn_key[0] == c_tuple[0] and conn_key[1] == c_tuple[1]) or \
               (conn_key[0] == c_tuple[1] and conn_key[1] == c_tuple[0]):
                mapped_color = [x / 255.0 for x in color_rgb] # RGB를 0-1 범위로 정규화
                break
        line_colors.append(mapped_color if mapped_color else [1.0, 1.0, 1.0]) # 기본값은 흰색

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(initial_landmarks), # 초기 점 데이터
        lines=o3d.utility.Vector2iVector(lines) # 연결 정보
    )
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    # Open3D 시각화 도구 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Pose Animation', width=1024, height=768)
    
    # 초기 지오메트리 (포인트 클라우드와 라인셋)를 뷰어에 추가
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    # 뷰 컨트롤 설정 (선택 사항: 초기 카메라 위치 및 줌 조정)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8) # 줌 레벨 조정
    # 초기 카메라 방향 설정 (예: 정면에서 바라보게)
    # ctr.set_front([0, 0, -1]) # Z 축 방향으로 앞을 바라봄
    # ctr.set_up([0, 1, 0])     # Y 축이 위를 향함
    # ctr.set_lookat([0, 0, 0]) # 원점을 중심으로 바라봄

    # 애니메이션 루프
    # 비디오의 FPS에 맞춰 프레임 간 지연 시간 설정
    delay_per_frame = 1.0 / fps if fps > 0 else 0.01 

    print("3D 시각화를 시작합니다. 창을 닫으면 종료됩니다.")
    while True:
        for i, frame_landmarks_3d in enumerate(all_frames_3d_landmarks):
            # 포인트 클라우드와 라인셋의 점 데이터를 현재 프레임의 랜드마크로 업데이트
            pcd.points = o3d.utility.Vector3dVector(frame_landmarks_3d)
            line_set.points = o3d.utility.Vector3dVector(frame_landmarks_3d) # 라인셋도 점 데이터를 업데이트해야 함
    
            # 업데이트된 지오메트리를 뷰어에 반영
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
            
            # 사용자 인터랙션 (마우스 회전, 확대/축소 등) 처리
            vis.poll_events()
            # 렌더링 업데이트 (화면을 다시 그림)
            vis.update_renderer()
    
            # 프레임 속도에 맞춰 잠시 대기
            time.sleep(delay_per_frame)
        input_value = input("Y/N")
        if input_value == "N" or input_value == "n":
            break

    vis.destroy_window() # 모든 프레임을 표시한 후 시각화 창 닫기
    print("3D 시각화가 종료되었습니다.")

def export_to_c3d(output_filename, all_frames_3d_landmarks, fps):
    if not all_frames_3d_landmarks:
        print("내보낼 3D 랜드마크 데이터가 없습니다.")
        return

    num_frames = len(all_frames_3d_landmarks)
    num_markers = all_frames_3d_landmarks[0].shape[0]

    points_data_all_frames = np.array(all_frames_3d_landmarks) * 1000.0 # meters to millimeters

    writer = c3d.Writer()

    for frame_idx in range(num_frames):
        current_frame_points_3d = points_data_all_frames[frame_idx]

        residuals_column = np.zeros((num_markers, 1), dtype=np.float32)

        points_with_residuals = np.hstack((current_frame_points_3d, residuals_column))
        points_with_residuals = np.hstack((points_with_residuals, residuals_column))

        writer.add_frames([(points_with_residuals, np.array([]))])

    with open(output_filename, 'wb') as handle:
        writer.write(handle)

    print(f"총 {num_frames} 프레임을 {output_filename}으로 성공적으로 내보냈습니다.")