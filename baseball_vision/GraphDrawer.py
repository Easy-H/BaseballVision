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

    def create_graph_image(self, cur_x:int, labels:list=None):
        if self.df is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        labels_to_plot = []
        if labels is None or not labels:
            labels_to_plot = [col for col in self.df.columns.tolist()]
        else:
            labels_to_plot = [label for label in labels if label in self.df.columns.tolist()]

        if self.df.empty or not labels_to_plot:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        min_val, max_val = 0, 180
        if labels_to_plot:
            relevant_data = self.df.iloc[:cur_x + 1][labels_to_plot].values
            min_val_calc = np.nanmin(relevant_data) if not np.all(np.isnan(relevant_data)) else np.nan
            max_val_calc = np.nanmax(relevant_data) if not np.all(np.isnan(relevant_data)) else np.nan
            
            if not pd.isna(min_val_calc) and not pd.isna(max_val_calc):
                min_val = min_val_calc
                max_val = max_val_calc

        y_ticks = np.arange(np.floor(min_val / 30) * 30, np.ceil(max_val / 30) * 30 + 1, 30)
        if len(y_ticks) < 2:
            if min_val == max_val:
                y_ticks = np.array([min_val - 30, min_val, min_val + 30])
            else:
                y_ticks = np.arange(min_val - 30, max_val + 30 + 1, 30)
                if len(y_ticks) < 2:
                    y_ticks = np.arange(np.floor(min_val/5)*5 - 30, np.ceil(max_val/5)*5 + 30 + 1, 30)

        self.ax.set_yticks(y_ticks)
        self.ax.set_ylim(min_val - 5, max_val + 5)
        
        # 모든 라인의 데이터를 초기화하여 숨김
        for label_name in self.lines.keys():
            self.lines[label_name].set_data([], [])
        
        # 모든 마커를 초기화하여 숨김
        for marker_obj in self.current_frame_markers.values():
            marker_obj.set_data([], [])

        # 선택된 라벨의 데이터만 업데이트
        data_to_plot_until_current_frame = self.df.iloc[:cur_x + 1]
        for label in labels_to_plot:
            if label in self.lines and label in data_to_plot_until_current_frame.columns:
                self.lines[label].set_data(data_to_plot_until_current_frame.index, data_to_plot_until_current_frame[label])
        
        # [수정 사항] 각 라벨에 대한 마커 업데이트
        if cur_x < len(self.df):
            for label in labels_to_plot:
                if label in self.df.columns and not pd.isna(self.df.loc[cur_x, label]):
                    marker_x = [cur_x]
                    marker_y = [self.df.loc[cur_x, label]]
                    
                    if label in self.current_frame_markers:
                        self.current_frame_markers[label].set_data(marker_x, marker_y)
                # 데이터가 없으면 마커를 숨깁니다.
                elif label in self.current_frame_markers:
                    self.current_frame_markers[label].set_data([], [])
        else:
            # 현재 프레임이 범위를 벗어나면 모든 마커를 숨깁니다.
            for marker_obj in self.current_frame_markers.values():
                marker_obj.set_data([], [])

        # 범례 업데이트
        if labels_to_plot:
            handles, plot_labels = [], []
            for lbl in labels_to_plot:
                if lbl in self.lines:
                    handles.append(self.lines[lbl])
                    plot_labels.append(lbl)

            if handles:
                self.ax.legend(handles=handles, labels=plot_labels,
                               loc='lower center',
                               bbox_to_anchor=(0.5, .85),
                               ncol=5,
                               markerscale=2,
                               fontsize=12, frameon=False,
                               bbox_transform=self.fig.transFigure)
            else:
                if self.ax.legend_ is not None:
                    self.ax.legend_.remove()
                    self.ax.legend_ = None
        else:
            if self.ax.legend_ is not None:
                self.ax.legend_.remove()
                self.ax.legend_ = None

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