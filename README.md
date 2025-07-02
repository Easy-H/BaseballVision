# BaseballVision
컴퓨터 비전 기술을 이용하여 야구 동작의 신체 정보를 수집하는 프로젝트

# 개발환경
- Jupyter
- Python 3.9
- Python Libraries
    - cv2
    - dlib
    - cmake
    - numpy as np
    - face_recognition
    - os
    - sys
    - mediapipe as mp
    - c3d
    - open3d
    - time
    - filterpy

# 개발사항
- MediaPipe를 사용한 영상 인물 Bone 작업
- Bone을 이용한 관절의 각도 계산
    - 팔꿈치 각도(손목-팔꿈치-어깨)
    - 어깨 각도(팔꿈치-어깨-어깨)
    - 상하체 꼬임 각도(어깨, 골반)
    - 무릎 각도(골반-무릎-발목)
    - 손목 각도(검지-손목-팔꿈치)

# Todo
- [ ] Bone 작업물 c3d 출력
- [ ] Bone 작업물 3D 영상 출력
- [ ] 기타 후처리 작업