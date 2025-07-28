import mediapipe as mp
import baseball_vision.angle_calc as ac
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # [수정 사항 1] Matplotlib 백엔드를 'Agg'로 설정
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image

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
        
        # [수정 사항 2] 모든 가능한 각도 라벨을 정의
        # 이는 calc_joints 메서드에서 반환될 수 있는 모든 각도의 이름 목록입니다.
        self.angle_history_labels = [
            "R Elbow", "L Elbow", "R Shoulder", "L Shoulder",
            "Shoulder", "Pelvis", "Body Twist", "R Knee", "L Knee",
            "R wrist", "L wrist" # BatterAnalysisTool에서 추가될 수 있는 라벨
        ]

        # [수정 사항 3] Figure, Axes, Line 객체들을 생성자에서 None으로 초기화
        # 이들은 _ensure_graph_initialized 메서드에서 한 번만 설정됩니다.
        self.fig = None
        self.ax = None
        self.lines = {}  # 각 라벨에 대한 Line2D 객체를 저장
        self.current_frame_marker = None

    def _ensure_graph_initialized(self, total_frames: int, width: int, height: int, 
                                  graph_title="Joint Angles Over Time", y_label="Degree"):
        """
        그래프 Figure와 Axes가 초기화되었는지 확인하고, 초기화되지 않았다면 설정합니다.
        이 메서드는 create_graph_image가 처음 호출될 때 실행되어 리소스 효율성을 높입니다.
        """
        if self.fig is None:
            # [수정 사항 4] Figure와 Axes를 한 번만 생성
            dpi = 100 # DPI는 이미지 품질과 Figure 크기에 영향을 줍니다.
            self.fig, self.ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
            
            self.ax.set_title(graph_title, fontsize=10)
            self.ax.set_xlabel("Frame Index", fontsize=8)
            self.ax.set_ylabel(y_label, fontsize=8)
            self.ax.tick_params(axis='both', which='major', labelsize=7)
            self.ax.grid(True, linestyle=':', alpha=0.6)
            
            # [수정 사항 5] 모든 가능한 라벨에 대해 빈 라인 객체를 미리 생성하고 저장
            # 이 객체들의 set_data() 메서드를 통해 매 프레임마다 데이터를 업데이트합니다.
            for label_name in self.angle_history_labels:
                line, = self.ax.plot([], [], label=label_name, linewidth=0.5)
                self.lines[label_name] = line
            
            # [수정 사항 6] 현재 프레임 마커 초기화
            self.current_frame_marker, = self.ax.plot([], [], 'o', markersize=4, color='red')
            
            # [수정 사항 7] X축 범위 초기 설정 (총 프레임 수 기반)
            self.ax.set_xlim(0, total_frames - 1)
            self.ax.set_ylim(0, 180) # 초기 Y축 기본값

            # 범례 공간 확보를 위해 tight_layout 대신 rect 사용
            # 범례 자체는 create_graph_image에서 동적으로 업데이트됩니다.
            self.fig.tight_layout(rect=[0, 0, 0.85, 1])

    def calc(self, landmarks):
        if landmarks is None:
            # 랜드마크가 None일 경우, 각도 값들을 NaN으로 채운 딕셔너리를 반환하여 df_results에 추가
            # 이렇게 해야 그래프가 중간에 끊기지 않고 NaN 값이 있는 곳은 선이 그려지지 않습니다.
            # df_results가 비어있을 때는 컬럼 정보가 없으므로 빈 딕셔너리로 시작
            nan_result = {col: np.nan for col in self.df_results.columns} if not self.df_results.empty else {}
            # 모든 angle_history_labels에 대해 NaN으로 채움 (새로운 프레임이 추가될 때 컬럼이 없을 경우 대비)
            for label in self.angle_history_labels:
                if label not in nan_result:
                    nan_result[label] = np.nan
                    
            self.df_results = pd.concat([self.df_results, pd.DataFrame([nan_result])], ignore_index=True)
            return nan_result
        
        result = self.calc_joints(get_joints(landmarks))
        self.df_results = pd.concat([self.df_results, pd.DataFrame([result])], ignore_index=True)
        return result
        
    def calc_joints(self, joints):
        pass
    
    def run(self):
        pass # 최종 작업을 수행할 수 있습니다. 예를 들어, self.close_graph() 호출 등
    
    def save(self, output_name):
        self.df_results.to_csv(output_name + ".csv", index=True)
        
    def get_dataframe(self):
        return self.df_results
    
    def show_graph(self, label=[]):
        # [show_graph 유지] 이 메서드는 새로운 Figure를 생성하여 한 번에 보여주는 용도로 사용됩니다.
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
        ax_show = plt.gca() # 현재 Axes를 가져옴
        
        for i in range(len(label)):
            ax_show.plot(self.df_results.index, self.df_results[label[i]], label=label[i])
        
        ax_show.set_title('Change in each joint angle (over time)', fontsize=16)
        ax_show.set_xlabel('frame idx', fontsize=12)
        ax_show.set_ylabel('degree', fontsize=12)
        
        # --- Y축 눈금을 15의 배수로 설정하는 부분 ---
        if not self.df_results.empty and label: 
            selected_data = self.df_results[label]
            min_val = selected_data.min().min()
            max_val = selected_data.max().max()

            y_ticks = np.arange(np.floor(min_val / 15) * 15,
                                np.ceil(max_val / 15) * 15 + 1,
                                15)
            ax_show.set_yticks(y_ticks) 
            ax_show.set_ylim(y_ticks.min() - 5, y_ticks.max() + 5) 

        ax_show.legend(loc='upper right')
        ax_show.grid(True)
        plt.tight_layout()
        plt.show()
        
    def create_graph_image(self, current_frame_idx: int, 
                           total_frames: int, width: int, height: int, labels: list = None,
                           graph_title="Joint Angles Over Time", y_label="Degree"):
        # [수정 사항 8] 그래프 요소들이 초기화되었는지 확인하고 필요하면 초기화합니다.
        self._ensure_graph_initialized(total_frames, width, height, graph_title, y_label)

        # [수정 사항 9] labels 파라미터 처리: 동적으로 플로팅할 라벨 결정
        labels_to_plot = []
        if labels is None: # labels가 None이면 focus_label 사용
            labels_to_plot = self.focus_label 
            if not labels_to_plot: # focus_label도 비어있으면 모든 df_results 컬럼 중 angle_history_labels에 있는 것만 사용
                labels_to_plot = [col for col in self.df_results.columns.tolist() if col in self.angle_history_labels]
        elif not labels: # labels 리스트가 비어있다면 df_results의 모든 컬럼 중 angle_history_labels에 있는 것만 사용
            labels_to_plot = [col for col in self.df_results.columns.tolist() if col in self.angle_history_labels]
        else: # labels가 명시적으로 주어진 경우 해당 라벨만 사용
            labels_to_plot = [label for label in labels if label in self.df_results.columns.tolist() and label in self.angle_history_labels]

        # df_results가 비어있거나 그릴 라벨이 없으면 빈 이미지 반환
        if self.df_results.empty or not labels_to_plot:
            return np.zeros((height, width, 3), dtype=np.uint8) 

        # [수정 사항 10] Y축 범위 및 눈금 계산: 현재까지의 데이터와 선택된 라벨 기반
        min_val, max_val = 0, 180 # 기본값 설정
        if not self.df_results.empty and labels_to_plot:
            # 현재까지의 모든 데이터 중, 선택된 라벨에 해당하는 데이터만 고려하여 Y축 범위 설정
            relevant_data = self.df_results[labels_to_plot].values
            # NaN 값은 무시하고 최소/최대값 찾기
            # 모든 값이 NaN인 경우를 대비하여 nanmin/nanmax 사용
            min_val_calc = np.nanmin(relevant_data) if not np.all(np.isnan(relevant_data)) else np.nan
            max_val_calc = np.nanmax(relevant_data) if not np.all(np.isnan(relevant_data)) else np.nan
            
            if not pd.isna(min_val_calc) and not pd.isna(max_val_calc):
                min_val = min_val_calc
                max_val = max_val_calc

        y_ticks = np.arange(np.floor(min_val / 15) * 15,
                            np.ceil(max_val / 15) * 15 + 1,
                            15)
        self.ax.set_yticks(y_ticks)
        self.ax.set_ylim(y_ticks.min() - 5, y_ticks.max() + 5)
        
        # [수정 사항 11] 모든 라인의 데이터를 초기화하여 숨김
        # 이는 이전에 그려진 라인들이 다음 프레임에서 사라지도록 합니다.
        for label_name in self.angle_history_labels:
            if label_name in self.lines:
                self.lines[label_name].set_data([], [])

        # [수정 사항 12] 선택된 라벨의 데이터만 업데이트 (set_data 사용)
        data_to_plot_until_current_frame = self.df_results.iloc[:current_frame_idx + 1]
        for label in labels_to_plot:
            if label in self.lines and label in data_to_plot_until_current_frame.columns:
                self.lines[label].set_data(data_to_plot_until_current_frame.index, data_to_plot_until_current_frame[label])
        
        # [수정 사항 13] 현재 프레임 마커 업데이트
        if current_frame_idx < len(self.df_results) and not self.df_results.empty:
            marker_x = [current_frame_idx]
            # 현재 프레임에서 플롯될 라인들의 유효한 Y값 평균을 마커 위치로 사용
            current_y_values = [self.df_results.loc[current_frame_idx, lbl] 
                                for lbl in labels_to_plot 
                                if lbl in self.df_results.columns and not pd.isna(self.df_results.loc[current_frame_idx, lbl])]
            
            if current_y_values:
                marker_y = np.mean(current_y_values)
            else:
                marker_y = self.ax.get_ylim()[0] # 데이터 없으면 Y축 하단에 표시
            self.current_frame_marker.set_data(marker_x, marker_y)
        else:
            self.current_frame_marker.set_data([], []) # 현재 프레임이 데이터 범위를 벗어나면 마커 숨김

        # [수정 사항 14] 범례 업데이트: 현재 표시되는 라벨에 대해서만 범례 갱신
        if labels_to_plot:
            # 현재 플롯된 라인들만 선택하여 범례를 새로 생성
            handles, plot_labels = [], []
            for lbl in labels_to_plot:
                if lbl in self.lines:
                    handles.append(self.lines[lbl])
                    plot_labels.append(lbl)
            if handles:
                self.ax.legend(handles=handles, labels=plot_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6, frameon=False)
            else:
                # 아무것도 플롯되지 않으면 범례 숨김
                if self.ax.legend_ is not None: # 기존 범례가 있으면 제거
                    self.ax.legend_.remove()
                    self.ax.legend_ = None
        else:
            if self.ax.legend_ is not None: # 기존 범례가 있으면 제거
                self.ax.legend_.remove()
                self.ax.legend_ = None

        # 그래프를 이미지로 변환
        buf = io.BytesIO() 
        self.fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_arr = np.array(Image.open(buf))
        buf.close()

        # RGBA를 BGR로 변환 (OpenCV 호환)
        if img_arr.shape[2] == 4:
            graph_image = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
        else:
            graph_image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            
        graph_image = cv2.resize(graph_image, (width, height))
        
        # [수정 사항 15] create_graph_image에서는 Figure를 닫지 않습니다.
        return graph_image
    
    def close_graph(self):
        """
        Matplotlib Figure를 명시적으로 닫아 메모리를 해제합니다.
        비디오 처리 완료 후 (예: PoseAnalysisProcessor의 run 메서드에서) 호출해야 합니다.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None # Figure를 닫은 후 다시 초기화될 수 있도록 설정
    def items(self):
        pass

class PitcherAnalysisTool(AnalysisTool):
    # [수정 사항 16] __init__ 메서드: 상위 클래스의 초기화를 호출하고, focus_label 기본값 설정
    def __init__(self, label:list=[]): 
        super().__init__(label) # AnalysisTool의 __init__ 호출
        if not label: # 전달된 label이 없으면 기본 focus_label 설정
            self.focus_label = ["R Elbow", "R Shoulder", "Shoulder", "Pelvis", "Body Twist", "R Knee"] 

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
    def items(self):
        return ["R Elbow", "L Elbow", "R Shoulder",
                "L Shoulder", "Shoulder", "Pelvis",
                "Body Twist", "R Knee", "L Knee"]

class BatterAnalysisTool(PitcherAnalysisTool):
    # [수정 사항 17] __init__ 메서드: 상위 클래스의 초기화를 호출하고, focus_label 기본값 설정
    def __init__(self, label:list=[]): 
        super().__init__(label) # PitcherAnalysisTool의 __init__ 호출
        if not label: # 전달된 label이 없으면 기본 focus_label 설정
            self.focus_label = ["R Elbow", "R Shoulder", "Shoulder", "Pelvis", "Body Twist", "R Knee"] # 타자 중점 각도

    def calc_joints(self, joints):
        result = super().calc_joints(joints)
        # 손목 각도 추가
        angle_R_wrist = ac.calculate_angle_3(
            joints["R_elbow"], joints["R_wrist"], joints["R_pinky_tip"])
        angle_L_wrist = ac.calculate_angle_3(
            joints["L_elbow"], joints["L_wrist"], joints["L_pinky_tip"])
        result["R wrist"] = angle_R_wrist
        result["L wrist"] = angle_L_wrist
        return result
    def items(self):
        ret = super.items()
        ret.append("R wrist")
        ret.append("L wrist")
        return ret