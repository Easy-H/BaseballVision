import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import io

class GraphDrawer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.lines = {}  # 각 라벨에 대한 Line2D 객체를 저장
        # 마커를 딕셔너리로 변경하여 각 라벨에 대한 마커를 관리합니다.
        self.current_frame_markers = {}

        self.df = None
        self.width = 0
        self.height = 0
        
    def show_graph(self, df, label):
        # show_graph 메서드는 변경 사항이 없습니다.
        plt.figure(figsize=(12, 8))
        ax_show = plt.gca()
        
        for i in range(len(label)):
            ax_show.plot(df.index, df[label[i]], label=label[i])
        
        ax_show.set_title('Change in each joint angle (over time)', fontsize=2)
        ax_show.set_xlabel('frame idx', fontsize=1)
        ax_show.set_ylabel('degree', fontsize=1)
        
        if not df.empty and label: 
            selected_data = df[label]
            min_val = selected_data.min().min()
            max_val = selected_data.max().max()

            y_ticks = np.arange(np.floor(min_val / 30) * 30, np.ceil(max_val / 30) * 30 + 1, 30)
            ax_show.set_yticks(y_ticks) 
            ax_show.set_ylim(y_ticks.min() - 5, y_ticks.max() + 5) 

        ax_show.legend(loc='lower center')
        ax_show.grid(True)

        plt.tight_layout()
        plt.show()

    def setting(self, df:pd.DataFrame, width:int, height:int,
                graph_title:str="", y_label:str=""):

        self.df = df
        self.width = width
        self.height = height

        dpi = 50
        self.fig, self.ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi * 2)

        self.ax.tick_params(axis='both', which='major', width="4", labelsize=16)
        self.ax.grid(True, linestyle=':', alpha=0.6)
            
        for label_name in self.df.columns.tolist():
            line, = self.ax.plot([], [], label=label_name, linewidth=4)
            self.lines[label_name] = line
            # 각 라벨에 대한 마커 객체를 미리 생성하여 저장합니다.
            marker, = self.ax.plot([], [], 'o', markersize=12, color=line.get_color())
            self.current_frame_markers[label_name] = marker
            
        self.ax.set_xlim(0, df.shape[0] - 1)
        self.ax.set_ylim(0, 180) 

        self.fig.tight_layout(rect=(0, 0, .97, .85))

    def create_graph_image(self, cur_x: int, labels: list = None):
        
        # 1. 초기 상태 및 데이터 유효성 검사
        if self.df is None or self.df.empty:
            print("DataFrame is None or empty.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 2. 플로팅할 라벨 목록 결정
        if labels is None or not labels:
            labels_to_plot = self.df.columns.tolist()
        else:
            labels_to_plot = [label for label in labels if label in self.df.columns]
        
        if not labels_to_plot:
            print("No valid labels to plot.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 3. 데이터 및 y축 범위 계산
        data_to_plot = self.df[labels_to_plot].iloc[:cur_x + 1]
        
        # relevant_data를 float 타입으로 변환
        relevant_data_float = data_to_plot.values.astype(float)

        # NaN 값 처리 및 min/max 값 계산
        if np.all(np.isnan(relevant_data_float)):
            min_val, max_val = 0, 180  # 기본값 설정
        else:
            min_val_calc = np.nanmin(relevant_data_float)
            max_val_calc = np.nanmax(relevant_data_float)
            min_val = min_val_calc if not pd.isna(min_val_calc) else 0
            max_val = max_val_calc if not pd.isna(max_val_calc) else 180
        
        y_ticks = calculate_y_ticks(min_val, max_val)
        # self.ax.set_yticks(y_ticks)
        # self.ax.set_ylim(y_ticks[0] - 5, y_ticks[-1] + 5) # 눈금 범위에 맞게 ylim 조정
        self.ax.set_yticks(y_ticks)
        self.ax.set_ylim(y_ticks[0], y_ticks[-1])
        
        # 5. 그래프 라인 및 마커 업데이트
        # 기존 코드와 동일한 로직을 사용하여 라인 및 마커 업데이트
        for label_name in self.lines.keys():
            self.lines[label_name].set_data([], [])
        
        for marker_obj in self.current_frame_markers.values():
            marker_obj.set_data([], [])

        for label in labels_to_plot:
            if label in self.lines:
                self.lines[label].set_data(data_to_plot.index, data_to_plot[label])
        
        if cur_x < len(self.df):
            for label in labels_to_plot:
                if label in data_to_plot.columns and not pd.isna(data_to_plot.loc[cur_x, label]):
                    if label in self.current_frame_markers:
                        self.current_frame_markers[label].set_data([cur_x], [data_to_plot.loc[cur_x, label]])
                elif label in self.current_frame_markers:
                    self.current_frame_markers[label].set_data([], [])
        else:
            for marker_obj in self.current_frame_markers.values():
                marker_obj.set_data([], [])

        # 6. 범례 업데이트
        # ... (기존 코드와 동일) ...
        handles, plot_labels = [], []
        for lbl in labels_to_plot:
            if lbl in self.lines:
                handles.append(self.lines[lbl])
                plot_labels.append(lbl)
        
        if handles:
            self.ax.legend(handles=handles, labels=plot_labels,
                        loc='lower center', bbox_to_anchor=(0.5, .85),
                        ncol=7, markerscale=2, handlelength=.2,
                        fontsize=12, frameon=False,
                        bbox_transform=self.fig.transFigure)
        else:
            if self.ax.legend_ is not None:
                self.ax.legend_.remove()
                self.ax.legend_ = None

        # 7. 이미지를 바이트 버퍼로 저장 및 반환
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', pad_inches=.2)
        buf.seek(0)
        img_arr = np.array(Image.open(buf))
        buf.close()

        if img_arr.shape[2] == 4:
            graph_image = img_arr[:, :, :3]
        else:
            graph_image = img_arr
        
        graph_image = cv2.resize(graph_image, (int(self.width), int(self.height)))
        
        return graph_image

    
    def close_graph(self):
        # close_graph 메서드는 변경 사항이 없습니다.
        if self.fig is None: return
        
        plt.close(self.fig)
        
        self.fig = None
        self.ax = None
        self.lines = {}
        self.current_frame_markers = {} # 마커 딕셔너리 초기화



def calculate_y_ticks(min_val: float, max_val: float, num_ticks: int = 5):
    """
    각도 데이터(0~180)의 최대/최소값에 따라 적절한 y축 눈금을 계산합니다.

    Args:
        min_val (float): 데이터의 최솟값.
        max_val (float): 데이터의 최댓값.
        num_ticks (int): 원하는 눈금의 개수 (기본값 5).

    Returns:
        np.ndarray: 계산된 y축 눈금 값들의 배열.
    """
    # 1. 예외 상황 처리: min_val과 max_val이 같을 경우
    if min_val == max_val:
        # 단일 값을 중심으로 3개의 눈금 배열 생성
        step = 30
        start = np.floor(min_val / step) * step
        y_ticks = np.array([start - step, start, start + step])
        return y_ticks

    # 2. 눈금 간격 (step size) 결정
    data_range = max_val - min_val
    approx_step = data_range / num_ticks

    angle_steps = [5, 10, 15, 20, 30, 45, 60, 90]
    
    step = 5
    for s in angle_steps:
        if s >= approx_step:
            step = s
            break

    # 3. 눈금 시작/끝 값 계산
    start = np.floor(min_val / step) * step
    end = np.ceil(max_val / step) * step
    
    # 4. y_ticks 생성 및 최소 눈금 개수 보장
    y_ticks = np.arange(start, end + 0.001, step)
    
    # 눈금 개수가 3개 미만일 경우, 간격을 더 줄여서 다시 계산
    if len(y_ticks) < 3:
        try:
            # 현재 step보다 작은 다음 step을 찾아 사용
            prev_step_idx = angle_steps.index(step) - 1
            if prev_step_idx >= 0:
                step = angle_steps[prev_step_idx]
            else:
                step = step / 2
        except ValueError:
            step = step / 2
        
        start = np.floor(min_val / step) * step
        end = np.ceil(max_val / step) * step
        y_ticks = np.arange(start, end + 0.001, step)

    # 최종적으로 y_ticks가 비어있을 경우, 기본값 반환
    if y_ticks.size == 0:
        return np.array([0, 90, 180])
        
    return y_ticks