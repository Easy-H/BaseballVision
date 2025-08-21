# graph_drawer.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import io
from utils.data.data_frame import IDataFrame

class GraphDrawer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.lines = {}
        self.current_frame_markers = {}
        self.df = None
        self.width = 0
        self.height = 0
        
    def show_graph(self, df:IDataFrame, label):
        plt.figure(figsize=(12, 8))
        ax_show = plt.gca()
        
        for i in range(len(label)):
            ax_show.plot(df.get_index(), df.get_data_column(label[i]), label=label[i])
        
        ax_show.set_title('Change in each joint angle (over time)', fontsize=20)
        ax_show.set_xlabel('frame idx', fontsize=15)
        ax_show.set_ylabel('degree', fontsize=15)
        
        if not df.is_empty() and label:
            min_val, max_val = df.get_max_min_values(label)
            
            y_ticks = np.arange(np.floor(min_val / 30) * 30, np.ceil(max_val / 30) * 30 + 1, 30)
            ax_show.set_yticks(y_ticks) 
            ax_show.set_ylim(y_ticks.min() - 5, y_ticks.max() + 5) 

        ax_show.legend(loc='lower center')
        ax_show.grid(True)
        plt.tight_layout()
        plt.show()

    def setting(self, df:IDataFrame, width:int, height:int, graph_title:str="", y_label:str=""):
        self.df = df
        self.width = width
        self.height = height

        dpi = 50
        self.fig, self.ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi * 2)

        self.ax.tick_params(axis='both', which='major', width="4", labelsize=16)
        self.ax.grid(True, linestyle=':', alpha=0.6)
            
        for label_name in self.df.get_column_list():
            line, = self.ax.plot([], [], label=label_name, linewidth=4)
            self.lines[label_name] = line
            marker, = self.ax.plot([], [], 'o', markersize=12, color=line.get_color())
            self.current_frame_markers[label_name] = marker
            
        self.ax.set_xlim(0, self.df.get_row_count() - 1)
        self.ax.set_ylim(0, 180) 
        self.fig.tight_layout(rect=(0, 0, .97, .85))

    def create_graph_image(self, cur_x: int, labels: list = None):
        if self.df is None or self.df.is_empty():
            print("DataFrame is None or empty.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if labels is None or not labels:
            labels_to_plot = self.df.get_column_list()
        else:
            labels_to_plot = [label for label in labels if label in self.df.get_column_list()]
        
        if not labels_to_plot:
            print("No valid labels to plot.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 현재 프레임까지의 데이터를 가져와서 y축 범위를 동적으로 설정
        data_to_plot_df = self.df.get_data_row_range((0, cur_x + 1))
        min_val, max_val = data_to_plot_df.get_max_min_values(labels_to_plot)

        y_ticks = calculate_y_ticks(min_val, max_val)
        self.ax.set_yticks(y_ticks)
        self.ax.set_ylim(y_ticks[0], y_ticks[-1])
        
        for label in self.lines.keys():
            if label in labels_to_plot:
                x_data = data_to_plot_df.get_index()
                y_data = data_to_plot_df.get_data_column(label)
                self.lines[label].set_data(x_data, y_data)
                
                if cur_x < self.df.get_row_count():
                    current_value = self.df.get_data_row(cur_x).get(label)
                    if current_value is not None:
                        self.current_frame_markers[label].set_data([cur_x], [current_value])
                    else:
                        self.current_frame_markers[label].set_data([], [])
            else:
                self.lines[label].set_data([], [])
                self.current_frame_markers[label].set_data([], [])
                
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
        if self.fig is None: return
        
        plt.close(self.fig)
        
        self.fig = None
        self.ax = None
        self.lines = {}
        self.current_frame_markers = {}

def calculate_y_ticks(min_val: float, max_val: float, num_ticks: int = 5):
    if min_val == max_val:
        step = 30
        start = np.floor(min_val / step) * step
        y_ticks = np.array([start - step, start, start + step])
        return y_ticks

    data_range = max_val - min_val
    approx_step = data_range / num_ticks

    angle_steps = [5, 10, 15, 20, 30, 45, 60, 90]
    
    step = 5
    for s in angle_steps:
        if s >= approx_step:
            step = s
            break

    start = np.floor(min_val / step) * step
    end = np.ceil(max_val / step) * step
    
    y_ticks = np.arange(start, end + 0.001, step)
    
    if len(y_ticks) < 3:
        try:
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

    if y_ticks.size == 0:
        return np.array([0, 90, 180])
        
    return y_ticks