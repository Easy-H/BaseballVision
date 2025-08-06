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
            
def update_visualization(vis_obj):
    """
    현재 프레임의 데이터로 Open3D 시각화를 업데이트합니다.
    """
    global current_frame_idx, all_frames_data, pcd, line_set, total_frames

    if current_frame_idx >= total_frames or current_frame_idx < 0:
        print(f"프레임 인덱스 {current_frame_idx}가 범위를 벗어났습니다 (0-{total_frames-1}). 경계를 조정합니다.")
        current_frame_idx = max(0, min(current_frame_idx, total_frames - 1))
        # 인덱스가 이미 범위를 벗어나 유효한 프레임으로 조정된 경우에도 업데이트해야 합니다.

    frame_landmarks_3d = all_frames_data[current_frame_idx]

    if frame_landmarks_3d is None:
        print(f"경고: 프레임 {current_frame_idx}의 랜드마크 데이터가 없습니다.")
        # 선택적으로 유효한 다음 프레임으로 건너뛸 수 있습니다.
        return False # 데이터 누락으로 인해 업데이트가 완전히 성공하지 못했음을 나타냄

    pcd.points = o3d.utility.Vector3dVector(frame_landmarks_3d)
    line_set.points = o3d.utility.Vector3dVector(frame_landmarks_3d)

    vis_obj.update_geometry(pcd)
    vis_obj.update_geometry(line_set)
    vis_obj.update_renderer()
    vis_obj.poll_events()
    return True # 업데이트 성공을 나타냄

def keyboard_callback(vis_obj, action, mods):
    """
    Callback function for keyboard events.
    'A': Previous frame
    'D': Next frame
    'S': Toggle play/pause
    """
    global current_frame_idx, total_frames, is_playing

    # Only react to key presses (action == 1)
    if action == 1:
        if vis_obj.get_key_press_value(ord('A')): # 'A' key for previous frame
            is_playing = False # Pause playback
            current_frame_idx -= 1
            print(f"Navigating to previous frame: {current_frame_idx}")
            update_visualization(vis_obj)
            return True
        elif vis_obj.get_key_press_value(ord('D')): # 'D' key for next frame
            is_playing = False # Pause playback
            current_frame_idx += 1
            print(f"Navigating to next frame: {current_frame_idx}")
            update_visualization(vis_obj)
            return True
        elif vis_obj.get_key_press_value(ord('S')): # 'S' key to toggle play/pause
            is_playing = not is_playing
            print(f"Playback toggled: {'Playing' if is_playing else 'Paused'}")
            return True
    return False # Return False if the event was not handled

def show_3d_video(landmarks_iterator, num_total_frames, fps):
    """
    수집된 3D 랜드마크 데이터를 Open3D를 사용하여 애니메이션으로 시각화합니다.
    키보드 'A'/'D' 키를 사용하여 프레임을 이동하고 'S' 키로 재생/일시정지를 토글할 수 있습니다.

    Args:
        landmarks_data_dir (str): 3D 랜드마크 데이터가 저장된 디렉토리 경로.
                                   각 프레임은 별도의 JSON 파일로 저장되어야 합니다.
        num_total_frames (int): 시각화할 총 프레임 수. (실제 파일 수와 다를 수 있음)
        fps (float): 시각화 속도를 조절하기 위한 초당 프레임 수.
    """
    global current_frame_idx, all_frames_data, vis, pcd, line_set, total_frames, is_playing
    total_frames = num_total_frames # 전역 total_frames 설정

    # 모든 데이터를 먼저 로드하여 초기화합니다.
    # _landmark_generator 함수가 이제 all_frames_data를 직접 채웁니다.
    # 로딩 프로세스를 트리거하기 위해 한 번 반복합니다.
    try:
        # 생성자를 소모하여 모든 데이터가 all_frames_data에 로드되도록 합니다.
        for _ in landmarks_iterator:
            pass
        
        if not all_frames_data or all_frames_data[0] is None:
            raise ValueError("로드 후 유효한 랜드마크 데이터를 찾을 수 없습니다.")
        
        initial_landmarks = all_frames_data[0]

    except (StopIteration, ValueError) as e:
        print(f"시각화를 초기화할 유효한 랜드마크 데이터를 찾을 수 없습니다: {e}")
        return

    # 포인트 클라우드 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(initial_landmarks)
    pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0] for _ in range(initial_landmarks.shape[0])]))

    # 라인셋 객체 생성
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

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(initial_landmarks),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    # Open3D 시각화 도구 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Pose Animation (A:이전, D:다음, S:재생/일시정지)', width=1024, height=768)
    
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    # 뷰 제어 설정 (선택 사항)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    # 키보드 콜백 등록
    vis.register_key_callback(ord('A'), keyboard_callback)
    vis.register_key_callback(ord('D'), keyboard_callback)
    vis.register_key_callback(ord('S'), keyboard_callback)

    # 애니메이션 루프
    delay_per_frame = 1.0 / fps if fps > 0 else 0.01 

    print("3D 시각화를 시작합니다. 창을 닫거나 'Esc' 키를 누르면 종료됩니다.")
    print("키보드 'A' (이전 프레임), 'D' (다음 프레임), 'S' (재생/일시정지)를 사용하세요.")

    while True:
        if is_playing:
            # 자동 재생을 위해 현재 프레임 인덱스 업데이트
            current_frame_idx = (current_frame_idx + 1) % total_frames
            
            # 업데이트가 실패하면 (예: 데이터 누락) 유효한 프레임이 나올 때까지 또는 데이터 끝까지 시도
            success = update_visualization(vis)
            while not success and current_frame_idx < total_frames -1:
                current_frame_idx = (current_frame_idx + 1) % total_frames
                success = update_visualization(vis)
                if current_frame_idx == 0: # 모든 후속 프레임이 누락된 경우 무한 루프 방지
                    break
            
            if not success and current_frame_idx == 0: # 루프를 돌았는데도 유효한 프레임을 찾을 수 없는 경우
                print("재생 모드에서 표시할 유효한 프레임을 찾을 수 없습니다.")
                is_playing = False # 재생 일시정지

        # 이벤트 처리 (키보드 입력 및 창 반응성에 중요)
        vis.poll_events()
        vis.update_renderer()

        # 창이 닫혔는지 확인
        if not vis.poll_events(): # poll_events가 창이 닫히면 False 반환
            break

        time.sleep(delay_per_frame)

    vis.destroy_window()
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