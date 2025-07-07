import open3d as o3d # Open3D 임포트
import config
import time
import numpy as np
import c3d # C3D library import
import os # 경로 조작을 위해 os 임포트

def _landmark_generator(landmarks_data_dir, num_frames):
    """
    디스크에서 프레임별로 3D 랜드마크 데이터를 생성하는 제너레이터입니다.
    """
    for i in range(num_frames):
        file_path = os.path.join(landmarks_data_dir, f"frame_{i:05d}.npy")
        if os.path.exists(file_path):
            yield np.load(file_path)
        else:
            print(f"경고: 랜드마크 파일을 찾을 수 없습니다: {file_path}. 프레임을 건너뜁니다.")
            yield None # 또는 적절하게 오류 처리

def show_3d_video(landmarks_data_dir, total_frames, fps):
    """
    수집된 3D 랜드마크 데이터를 Open3D를 사용하여 애니메이션으로 시각화합니다.

    Args:
        all_frames_3d_landmarks (list): 각 프레임의 3D 랜드마크 (Numpy 배열 리스트).
                                        각 배열은 (num_markers, 3) 형태. (미터 단위)
        connections (list): 랜드마크 연결 정보를 담은 튜플 리스트 (예: [(0, 1), (1, 2)]).
                            MediaPipe의 PoseLandmark.value 인덱스를 사용.
        fps (float): 시각화 속도를 조절하기 위한 초당 프레임 수.
    """
    if not os.path.isdir(landmarks_data_dir) or total_frames == 0:
        print("시각화할 3D 랜드마크 데이터가 없습니다.")
        return

    landmarks_iterator = _landmark_generator(landmarks_data_dir, total_frames)

    # 초기화를 위해 첫 프레임 가져오기
    try:
        initial_landmarks = next(landmarks_iterator)
        while initial_landmarks is None: # 처음 몇 프레임이 누락된 경우 건너뛰기
            initial_landmarks = next(landmarks_iterator)
    except StopIteration:
        print("시각화를 초기화할 유효한 랜드마크 데이터를 찾을 수 없습니다.")
        return

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

def export_to_c3d(output_filename, landmarks_data_dir, total_frames, fps):
    if not os.path.isdir(landmarks_data_dir) or total_frames == 0:
        print("내보낼 3D 랜드마크 데이터가 없습니다.")
        return

    num_frames = total_frames # 프로세서에서 전달된 total_frames 사용
    # markers 수는 첫 프레임 또는 설정에서 파생될 수 있다고 가정합니다.
    # 현재는 markers 수를 얻기 위해 첫 프레임을 로드합니다.
    first_frame_path = os.path.join(landmarks_data_dir, "frame_00000.npy")
    if not os.path.exists(first_frame_path):
        print(f"오류: 첫 랜드마크 파일 {first_frame_path}을(를) 찾을 수 없습니다.")
        return
    num_markers = np.load(first_frame_path).shape[0]

    writer = c3d.Writer()

    # 제너레이터를 사용하여 반복
    for frame_idx, current_frame_points_3d in enumerate(_landmark_generator(landmarks_data_dir, total_frames)):
        if current_frame_points_3d is None:
            # 필요한 경우 C3D 내보내기에서 누락된 프레임 처리 (예: 0을 쓰거나 건너뛰기)
            # 단순화를 위해 일관된 수의 마커를 가정하고 누락된 경우 NaN으로 채웁니다.
            # 또는 제너레이터가 유효한 데이터를 반환하거나 오류를 발생시키도록 처리해야 합니다.
            # 프레임이 실제로 누락된 경우 0 또는 특수 값을 쓰고 싶을 수 있습니다.
            # 현재는 None인 경우 건너뛰거나 NaN으로 채워 프레임 수를 유지합니다.
            print(f"경고: C3D 내보내기 중 프레임 {frame_idx} 데이터가 누락되었습니다. NaN으로 채웁니다.")
            current_frame_points_3d = np.full((num_markers, 3), np.nan) # 누락된 데이터에 대해 NaN으로 채우기

        points_data_mm = current_frame_points_3d * 1000.0 # 미터에서 밀리미터로

        residuals_column = np.zeros((num_markers, 1), dtype=np.float32)

        points_with_residuals = np.hstack((points_data_mm, residuals_column))
        points_with_residuals = np.hstack((points_with_residuals, residuals_column))

        writer.add_frames([(points_with_residuals, np.array([]))])

    with open(output_filename, 'wb') as handle:
        writer.write(handle)

    print(f"총 {num_frames} 프레임을 {output_filename}으로 성공적으로 내보냈습니다.")