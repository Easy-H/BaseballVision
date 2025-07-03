# ⚾️ BaseballVision
- 컴퓨터 비전 기술을 이용하여 야구 동작(투구, 타격)의 신체 정보를 수집하는 프로젝트입니다.
- 이 프로젝트는 컴퓨터 비전 기술을 활용하여 야구 선수들의 동작을 정밀하게 분석하고, 이를 통해 개인의 스윙/투구 폼 개선, 잠재력 향상을 돕는 것을 목표로 합니다.

## 🎉 결과물 (Results)
### ✍🏻 입력 (Input)
#### 원본 영상
![원본 영상](readme_image/이지헌.gif)

### 🖨️ 출력 (Output)
#### 신체 본 영상
![신체 본 영상](readme_image/이지헌_bone_output.gif)
#### 신체 본, 원본 합성 영상
![신체 본, 원본 합성 영상](readme_image/이지헌_combined_output.gif)

## 🔧 개발사항 (Developments)
- MediaPipe를 사용한 영상 인물 Bone 작업: "영상 속 야구 선수의 2D/3D 핵심 관절(Landmarks)을 정밀하게 추출하고, 이를 기반으로 인체의 골격(Bone) 모델을 생성합니다. 이를 통해 움직임을 디지털화합니다."
- Bone을 이용한 관절의 각도 계산: 추출된 Bone 데이터를 활용하여 야구 동작의 주요 지표가 되는 관절 각도를 계산합니다. 이는 선수 폼 분석의 핵심 지표로 활용될 수 있습니다.
    - 팔꿈치 각도(손목-팔꿈치-어깨): 투구 또는 타격 시 팔꿈치의 굽힘 정도를 분석하여 효율적인 힘 전달 여부를 판단할 수 있습니다.
    - 어깨 각도(팔꿈치-어깨): 회전 운동에서 어깨의 가동 범위 및 안정성을 평가할 수 있습니다.
    - 상하체 꼬임 각도(어깨, 골반): 타격/투구 시 상하체 분리(Torque) 정도를 정량화하여 파워 생성 능력을 측정할 수 있습니다.
    - 무릎 각도(골반-무릎-발목): 하체 지면 반력 활용 및 동작 안정성에 기여하는 무릎의 움직임을 분석할 수 있습니다.
    - 손목 각도(검지-손목-팔꿈치): "릴리스/임팩트 순간의 손목 각도를 통해 공의 회전 또는 타구 방향에 미치는 영향을 분석할 수 있습니다.

### 📋 할일 (Todo)
- [ ] Bone 작업물 c3d 출력
- [ ] Bone 작업물 3D 영상 출력
- [ ] 기타 후처리 작업

## 🛠️ 개발 환경 (Development Environment)

- **언어:** Python 3.9
- **통합 개발 환경 (IDE):** Jupyter Notebook
- **주요 라이브러리:**
    - `mediapipe`: 고성능 인체 자세 추정(Pose Estimation) 및 3D 랜드마크 추출
    - `opencv-python` (`cv2`): 영상 데이터 로딩, 전처리 및 시각화
    - `numpy`: 랜드마크 데이터 및 각도 계산을 위한 수치 연산
    - `pandas`: 데이터 정리하는 자료
    - `Matplotlib`: 데이터 시각화
    - `open3d`: 3D 포즈 데이터 및 관절 시각화 (동작 3D 영상 출력)
    - `c3d`: 3D 모션 데이터 표준 포맷인 C3D 파일 처리 (모션 데이터 출력)
    - `filterpy`: (예: 관절 각도 데이터의 노이즈 제거 및 평활화를 위한 칼만 필터 등)
    - `dlib`, `cmake`, `os`, `sys`, `time`: 기타 유틸리티에 사용

## 🚀 사용 방법 (How to Run)

1.  **저장소 클론:**
    ```bash
    git clone [https://github.com/YourUsername/BaseballVision.git](https://github.com/Easy-H/BaseballVision.git)
    cd BaseballVision
    ```
2.  **가상 환경 설정 (권장):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Jupyter Notebook 실행:**
    ```bash
    jupyter notebook
    ```
5.  `baseball_pose_analysis.ipynb` 파일을 열어 셀을 실행합니다.
