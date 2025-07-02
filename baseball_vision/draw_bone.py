import cv2
import mediapipe as mp
import open3d as o3d # Open3D 임포트
import numpy as np
import config
import time

def landmarks_animation(all_frames_3d_landmarks, connections, fps):
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
    for c_tuple in connections:
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
        if input("Y/N") == "N":
            break
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

    vis.destroy_window() # 모든 프레임을 표시한 후 시각화 창 닫기
    print("3D 시각화가 종료되었습니다.")


def draw_landmarks_custom(image, landmarks, image_width, image_height):
    """
    Draws custom-styled pose landmarks on the image.
    - Simplifies face to a single nose node.
    - Draws other body points as small gray dots.

    Args:
        image (np.array): The OpenCV BGR image frame to draw on.
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                    The pose landmarks detected by MediaPipe.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    for idx, landmark in enumerate(landmarks.landmark):
        # Only draw if landmark visibility is good
        if landmark.visibility < config.MIN_DRAW_VISIBILITY:
            continue
        
        center_coordinates = (int(landmark.x * image_width), int(landmark.y * image_height))

        if idx == mp.solutions.pose.PoseLandmark.NOSE.value: # Nose: single face node
            cv2.circle(image, center_coordinates, 5, (255, 255, 255), -1) # White circle
        elif 1 <= idx <= 10: # Other facial landmarks (eyes, ears, mouth): don't draw
            pass
        else: # Body, arm, leg landmarks: small gray dot
            cv2.circle(image, center_coordinates, 2, (100, 100, 100), -1)

def draw_connections_custom(image, landmarks, image_width, image_height):
    """
    Draws custom color-coded pose connections (bones) on the image.

    Args:
        image (np.array): The OpenCV BGR image frame to draw on.
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                    The pose landmarks detected by MediaPipe.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    for connection in mp.solutions.pose.POSE_CONNECTIONS:
        idx1, idx2 = connection
        
        if landmarks.landmark[idx1].visibility < config.MIN_DRAW_VISIBILITY or landmarks.landmark[idx2].visibility < config.MIN_DRAW_VISIBILITY:
            continue
        
        # Get color for the connection from the predefined map
        color = config.CONNECTIONS_COLORS.get(connection, None)
        if color is None: # Check if tuple order is reversed in map
            color = config.CONNECTIONS_COLORS.get((idx2, idx1), None)

        if color is not None:
            point1 = (int(landmarks.landmark[idx1].x * image_width), int(landmarks.landmark[idx1].y * image_height))
            point2 = (int(landmarks.landmark[idx2].x * image_width), int(landmarks.landmark[idx2].y * image_height))
            cv2.line(image, point1, point2, color, 2) # Line thickness 2

def draw_pose_on_frame(frame, pose_landmarks):
    """
    Orchestrates drawing pose landmarks and connections on a frame with custom styling.

    Args:
        frame (np.array): The OpenCV BGR image frame.
        pose_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                        The pose landmarks detected by MediaPipe.

    Returns:
        np.array: The frame with the custom-drawn pose.
    """
    frame_with_pose = frame.copy()
    h, w, _ = frame.shape
    
    # Call functions to draw landmarks and connections
    draw_landmarks_custom(frame_with_pose, pose_landmarks, w, h)
    draw_connections_custom(frame_with_pose, pose_landmarks, w, h)
    
    return frame_with_pose