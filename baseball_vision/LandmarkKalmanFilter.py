from filterpy.kalman import KalmanFilter
import numpy as np

class LandmarkKalmanFilter:
    def __init__(self, num_landmarks=33, dim_state=6, dim_measurement=3, dt=1.0,
                 process_noise_std=0.01, measurement_noise_std=0.1):
        # dim_state: 상태 벡터의 차원 (x, y, z, vx, vy, vz) = 6
        # dim_measurement: 측정 벡터의 차원 (x, y, z) = 3
        # dt: 시간 간격 (여기서는 프레임 간 간격, 1로 가정)
        
        self.filters = []
        for _ in range(num_landmarks):
            kf = KalmanFilter(dim_x=dim_state, dim_z=dim_measurement)

            # 상태 전이 행렬 (F): x = x + vx*dt, y = y + vy*dt, z = z + vz*dt
            kf.F = np.array([[1, 0, 0, dt, 0, 0],
                             [0, 1, 0, 0, dt, 0],
                             [0, 0, 1, 0, 0, dt],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])

            # 측정 행렬 (H): 측정값은 상태 벡터의 (x, y, z) 부분
            kf.H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0]])

            # 공분산 행렬 (P): 초기 상태 불확실성 (클수록 초기 수렴 빠름)
            kf.P *= 1000.

            # 프로세스 노이즈 공분산 (Q): 모델 예측의 불확실성 (높을수록 필터가 측정값에 더 민감)
            # 랜드마크 움직임의 불확실성이 크다면 높게 설정 (예: 0.01)
            kf.Q = np.diag([process_noise_std**2]*3 + [process_noise_std**2]*3) # x,y,z 및 vx,vy,vz에 대한 노이즈

            # 측정 노이즈 공분산 (R): 측정값의 불확실성 (높을수록 필터가 측정값을 덜 신뢰)
            # MediaPipe 랜드마크의 노이즈가 심하면 높게 설정 (예: 0.1)
            kf.R = np.diag([measurement_noise_std**2]*3) # x,y,z 측정 노이즈

            # 초기 상태 벡터 (x): [x, y, z, vx, vy, vz]
            kf.x = np.zeros((dim_state, 1)) # 초기값은 첫 측정값으로 설정

            self.filters.append(kf)
        self.initialized = False # 필터 초기화 여부 플래그

    def initialize_state(self, first_landmarks_coords):
        # 첫 프레임에서 모든 랜드마크의 초기 상태를 설정 (측정값으로)
        for i, kf in enumerate(self.filters):
            if i < len(first_landmarks_coords): # 유효한 랜드마크 개수 범위 내에서
                # x, y, z는 첫 측정값으로, 속도(vx,vy,vz)는 0으로 초기화
                kf.x = np.array([[first_landmarks_coords[i][0]],
                                 [first_landmarks_coords[i][1]],
                                 [first_landmarks_coords[i][2]],
                                 [0.], [0.], [0.]])
        self.initialized = True

    def filter(self, current_landmarks_coords, visibility_scores=None, min_visibility_threshold=0.5):
        # current_landmarks_coords: 현재 프레임의 (num_landmarks, 3) numpy 배열
        # visibility_scores: 현재 프레임의 (num_landmarks,) numpy 배열 (0~1)
        
        filtered_coords = np.zeros_like(current_landmarks_coords)
        for i, kf in enumerate(self.filters):
            # predict 단계
            kf.predict()
            
            # update 단계: visibility 점수 기반으로 측정값 사용 여부 결정
            if visibility_scores is not None and visibility_scores[i] < min_visibility_threshold:
                # 랜드마크가 충분히 보이지 않으면 측정값을 사용하지 않고 예측값만 사용
                filtered_coords[i] = kf.x[:3].flatten() # 상태 벡터의 위치 부분만 사용
            else:
                # 랜드마크가 잘 보이면 측정값으로 업데이트
                kf.update(current_landmarks_coords[i].reshape(-1, 1))
                filtered_coords[i] = kf.x[:3].flatten()

        return filtered_coords