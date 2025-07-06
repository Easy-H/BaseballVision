import mediapipe as mp
import baseball_vision.angle_calc as ac
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_joints(landmarks):
     return { "R_shoulder": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
                "R_elbow": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
                "R_wrist": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
                "R_hip": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP],
                "R_knee": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE],
                "R_ankle": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE],
                "L_shoulder": landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
                "L_elbow": landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
                "L_wrist": landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST],
                "L_hip": landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP],
                "L_knee": landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE],
                "L_ankle": landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE],
                "R_pinky_tip": landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX],
                "L_pinky_tip": landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]}
    
class AnalysisTool:
    def __init__(self, label:list):
        self.df_results = pd.DataFrame()
        self.focus_label = label
    def calc(self, landmarks):
        if landmarks is None:
            self.df_results = pd.concat([self.df_results, pd.DataFrame([{}])], ignore_index=True)
            return {}
        result = self.calc_joints(get_joints(landmarks))
        self.df_results = pd.concat([self.df_results, pd.DataFrame([result])], ignore_index=True)
        return result
    def calc_joints(self, joints):
        pass
    def run(self):
        pass
    def save(self, output_name):
        self.df_results.to_csv(output_name + ".csv", index=True)
    def get_dataframe(self):
        return self.df_results
    def show_graph(self, label=[]):
        remove_idx = []
        for i in range(len(label)):
            if not label[i] in self.df_results.columns.tolist():
                print(label[i] + "is not correct name")
                remove_idx.append(i)
        remove_idx.reverse()
        for i in range(len(remove_idx)):
            del label[remove_idx[i]]
            
        if len(label)==0:
            label = self.df_results.columns.tolist()
        plt.figure(figsize=(12, 8))

        for i in range(len(label)):
            plt.plot(self.df_results.index, self.df_results[label[i]], label=label[i])
        
        plt.title('Change in each joint angle (over time)', fontsize=16)
        plt.xlabel('frame idx', fontsize=12)
        plt.ylabel('degree', fontsize=12)
        
        # --- Y축 눈금을 15의 배수로 설정하는 부분 추가 시작 ---
        if not self.df_results.empty and label: # df_results가 비어있지 않고, 그릴 라벨이 있을 때만 실행
            # 선택된 라벨들의 데이터 범위 계산
            selected_data = self.df_results[label]
            min_val = selected_data.min().min()
            max_val = selected_data.max().max()

            # 15의 배수로 Y축 눈금 생성
            y_ticks = np.arange(np.floor(min_val / 15) * 15,
                                np.ceil(max_val / 15) * 15 + 1,
                                15)
            plt.yticks(y_ticks) # plt.yticks()를 사용하여 현재 활성화된 Axes의 Y축 눈금을 설정합니다.
            # Y축 범위도 눈금에 맞춰 설정 (선택 사항이지만 권장)
            plt.ylim(y_ticks.min() - 5, y_ticks.max() + 5) # 눈금보다 약간 더 넓게 범위 설정
        # --- Y축 눈금을 15의 배수로 설정하는 부분 추가 끝 ---

        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def create_graph_image(self, current_frame_idx: int, 
                           total_frames: int, width, height, labels: list = None,
                           graph_title="Joint Angles Over Time", y_label="Degree"):
        if labels is None:
            labels = self.focus_label
        if len(labels)==0:
            labels = self.df_results.columns.tolist()
        # df_results가 비어있으면 빈 이미지 반환 또는 오류 처리
        if self.df_results.empty:
            return np.zeros((height, width, 3), dtype=np.uint8) # 예시 크기, 검정색 이미지

        dpi = 100 # 그래프 해상도 (조절 가능)
        fig_width = width / dpi
        fig_height = height / dpi
        
        fig, ax = plt.subplots(figsize=(6, 2), dpi=100) 
        
        min_val = self.df_results[labels].min().min()
        max_val = self.df_results[labels].max().max()

        # 15의 배수로 Y축 눈금 생성
        # np.arange(start, stop, step)을 사용하여 눈금 배열을 만듭니다.
        # 시작점은 min_val보다 작거나 같은 15의 배수 중 가장 큰 값
        # 끝점은 max_val보다 크거나 같은 15의 배수 중 가장 작은 값
        y_ticks = np.arange(np.floor(min_val / 15) * 15,
                            np.ceil(max_val / 15) * 15 + 1, # +1은 상한값을 포함하기 위함
                            15)
        ax.set_yticks(y_ticks)
        # 필요에 따라 Y축 범위도 함께 설정하여 눈금이 잘 보이도록 합니다.
        ax.set_ylim(y_ticks.min() - 5, y_ticks.max() + 5) # 눈금보다 약간 더 넓게 범위 설정
        
        # 현재 프레임까지의 데이터만 사용
        data_to_plot = self.df_results.iloc[:current_frame_idx + 1]
    
        for label in labels:
            if label in data_to_plot.columns:
                ax.plot(data_to_plot.index, data_to_plot[label], label=label, linewidth=.5)
    
        # 현재 프레임 위치 강조 (마지막 데이터 포인트)
        if current_frame_idx < len(self.df_results) and not self.df_results.empty:
            for label in labels:
                if label in self.df_results.columns:
                    # 마지막 유효한 데이터 포인트에 마커 표시
                    if current_frame_idx < len(self.df_results[label]): # Ensure index exists
                        ax.plot(current_frame_idx, self.df_results[label].iloc[current_frame_idx], 'o', markersize=4, color='red')
    
        ax.set_title(graph_title, fontsize=10)
        ax.set_xlabel("Frame Index", fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        
        # X축 범위 고정 (전체 영상 길이에 맞춰)
        ax.set_xlim(0, total_frames - 1)
        
        # Y축 범위는 데이터에 따라 동적으로 설정하거나, 적절한 고정 값 사용 (예: ax.set_ylim(-180, 180))
        # ax.set_ylim(df_results[labels].min().min() - 10, df_results[labels].max().max() + 10) # 데이터 기반 동적 범위
    
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6, frameon=False) # 범례 위치 조정
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # 범례 공간 확보
        
        # 그래프 이미지 변환 후 리턴
        fig.canvas.draw()
        actual_width, actual_height = fig.canvas.get_width_height()
            
        graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # 가져온 실제 폭과 높이를 사용하여 reshape합니다.
        graph_image = graph_image.reshape(actual_height, actual_width, 3) 
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)

        graph_image = cv2.resize(graph_image, (width, height))
        
        plt.close(fig)
        return graph_image

class PitcherAnalysisTool(AnalysisTool):
    def calc_joints(self, joints):
        # --- 각도 계산 ---
        # 1. 오른쪽 팔꿈치 각도
        angle_R_elbow = ac.calculate_angle_3(
            joints["R_shoulder"], joints["R_elbow"], joints["R_wrist"])
        angle_L_elbow = ac.calculate_angle_3(
            joints["L_shoulder"], joints["L_elbow"], joints["L_wrist"])
    
        # 2. 오른쪽 어깨 각도 (몸통-어깨-팔꿈치"])
        angle_R_shoulder = ac.calculate_angle_3(
            joints["L_shoulder"], joints["R_shoulder"], joints["R_elbow"])
        angle_L_shoulder = ac.calculate_angle_3(
            joints["R_shoulder"], joints["L_shoulder"], joints["L_elbow"])

        angle_shoulder = ac.calculate_angle_4(
            joints["R_shoulder"], joints["L_shoulder"], np.array([0, 0, 0]), np.array([10, 0, 0])
        )
        angle_pelvis = ac.calculate_angle_4(
            joints["R_hip"], joints["L_hip"], np.array([0, 0, 0]), np.array([10, 0, 0])
        )
        # 3. 오른쪽 골반 각도 (몸통-골반-무릎"])
        angle_body_twist = angle_shoulder - angle_pelvis
        angle_body_twist = (angle_body_twist + 180) % 360 - 180
        
        # 4. 오른쪽 무릎 각도
        angle_R_knee = ac.calculate_angle_3(
            joints["R_hip"], joints["R_knee"], joints["R_ankle"])
        angle_L_knee = ac.calculate_angle_3(
            joints["L_hip"], joints["L_knee"], joints["L_ankle"])

        return { "R Elbow": round(angle_R_elbow, 2),
                "L Elbow": round(angle_L_elbow, 2),
                "R Shoulder": round(angle_R_shoulder, 2),
                "L Shoulder": round(angle_L_shoulder, 2),
                "Shoulder": round(angle_shoulder, 2),
                "Pelvis": round(angle_pelvis, 2),
                "Body Twist": round(angle_body_twist, 2),
                "R Knee": round(angle_R_knee, 2),
                "L Knee": round(angle_L_knee, 2)}

class BatterAnalysisTool(PitcherAnalysisTool):
    def calc_joints(self, joints):
        result = super().calc_joints(joints)
        angle_R_wrist = ac.calculate_angle_3(
            joints["R_elbow"], joints["R_wrist"], joints["R_pinky_tip"])
        angle_L_wrist = ac.calculate_angle_3(
            joints["L_elbow"], joints["L_wrist"], joints["L_pinky_tip"])
        result["R wrist"] = angle_R_wrist
        result["L wrist"] = angle_L_wrist
        return result
    