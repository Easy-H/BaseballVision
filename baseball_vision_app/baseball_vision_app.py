from .process_setting_app import ProcessSetting
from .graph_draw_setting_app import GraphSetting
from .process_app import ProcessApp

from baseball_vision.pose.frame_maker \
    import PoseFrameMaker, VConcatFrameMaker,\
        GraphFrameMaker, PoseOverlayFrameMaker, PoseOnlyFrameMaker,\
        TraceFrameMaker

import baseball_vision.pose.analysis_tool as bvtool
import baseball_vision.pose.visualizer.draw_3d as d3d
from baseball_vision import ProcessedData

from gui.WidgetBuilder import WidgetBuilder

import utils
from utils import VideoMaker

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk # Pillow 라이브러리
import platform

def bool_to_(b):
    if b:
        return tk.NORMAL
    return tk.DISABLED

class BaseballVisionApp(WidgetBuilder):

    def __init__(self, master):
        #super().__init__()
        self.master = master
        master.overrideredirect(False)
        master.focus_set()

        data = utils.yaml.load_yaml("Layout\BaseballVisionApp.yaml")
        self.set_window(master, data)
        self.create_widget(master, data)

        self._after_id = None
        master.bind("<Configure>", self.on_window_resize)

        self.pose_frame_maker_dict = {
            "Basic": PoseFrameMaker(),
            "PoseOnly": PoseOnlyFrameMaker(),
            "BasicOverlayData": PoseOverlayFrameMaker(PoseFrameMaker()),
            "PoseOnlyOverlayData": PoseOverlayFrameMaker(PoseOnlyFrameMaker()),
            "Trace": TraceFrameMaker()
        }

        self.data_frame_maker_dict = {
            "Graph": GraphFrameMaker()
        }

        self.analysis_tool_dict = {
            "Joint Angle": bvtool.JointAnalysisTool(),
            "Velocity": bvtool.VelocityAnalysisTool(),
            "Height": bvtool.HeightAnalysisTool()
        }

        self._bind_method()

        self.bv_data = None
        self.df = None

        self.tool = bvtool.JointAnalysisTool()
        #self.tool = bvtool.VelocityAnalysisTool()

        self.visualize_parameter = []

        self.video_frame_maker = self.set_video_frame_maker()

        self.video_maker = VideoMaker()

        self.is_video_play = False
        self.widget_config("btn_play_video", "text", "▶")

        self._update_output(f"환경:{ platform.architecture() }")
        self._update_output("준비되었습니다. '비디오 선택' 버튼을 눌러주세요.\n")
    
    def on_window_resize(self, event):
        if self._after_id:
            self.master.after_cancel(self._after_id)  # Cancel previous scheduled action
            
        self._after_id = self.master.after(10, self.on_resize_finished)
        
    def on_resize_finished(self):
        self.show_img()

    def _bind_method(self):
        self.widget_config("btn_process_video", "command", self.process_video)
        self.widget_config("btn_setting", "command", self.open_settings)
        self.widget_config("btn_graph_setting", "command", self.open_graph_settings)
        self.widget_config("btn_play_3d_video", "command", self.show_3d_video)
        self.widget_config("btn_play_video", "command", self.play_video)
        self.widget_config("btn_load_data", "command", self.load_data)
        self.widget_config("scale_get_idx", "command", self.show_img)
        self.widget_config("btn_save_video", "command", self.save_video)
        self.widget_config("btn_save_data", "command", self.save_data)
        self.widget_config("btn_save_csv", "command", self.save_csv)
       
        self.widget_config("combobox_pose_frame_maker",
                           "values", [*self.pose_frame_maker_dict.keys()])
        self.widget_config("combobox_data_frame_maker",
                           "values", [*self.data_frame_maker_dict.keys()])
        self.widget_config("combobox_analysis_tool",
                           "values", [*self.analysis_tool_dict.keys()])
        
        self.objs["combobox_pose_frame_maker"].current(2) 
        self.objs["combobox_data_frame_maker"].current(0) 
        self.objs["combobox_analysis_tool"].current(0) 

        self.objs["combobox_pose_frame_maker"].bind(
            "<<ComboboxSelected>>", self.frame_maker_setting_changed)
        self.objs["combobox_data_frame_maker"].bind(
            "<<ComboboxSelected>>", self.frame_maker_setting_changed)
        self.objs["combobox_analysis_tool"].bind(
            "<<ComboboxSelected>>", self.analysis_tool_setting_changed)
        
        self.objs["scale_get_idx"].bind("<Button-1>", self.pause_video)
    
    def frame_maker_setting_changed(self, event=None):
        self.set_video_frame_maker()
        self.show_img()
    
    def analysis_tool_setting_changed(self, event=None):

        self.df = self.get_analysis_tool().calc(self.bv_data)
        self.set_video_frame_maker()
        self.show_img()

    def set_video_frame_maker(self):
        if self.bv_data is None:
            return
        
        pose_idx = "BasicOverlayData"
        data_idx = "Graph"
        
        if "combobox_pose_frame_maker" in self.objs:
            pose_idx = self.objs["combobox_pose_frame_maker"].get()
        if "combobox_data_frame_maker" in self.objs:
            data_idx = self.objs["combobox_data_frame_maker"].get()

        pose_frame_maker = self.pose_frame_maker_dict[pose_idx]
        data_frame_maker = self.data_frame_maker_dict[data_idx]

        pose_frame_maker.set_data(self.bv_data, self.df)
        pose_frame_maker.set_focus_label(self.visualize_parameter)
        data_frame_maker.set_graph(self.df,
            (int(self.bv_data.raw_video_width_list[0]), 200))
        data_frame_maker.set_focus_label(self.visualize_parameter)

        self.video_frame_maker= VConcatFrameMaker([pose_frame_maker,
                                                   data_frame_maker])
    
    def play_video(self):
        if self.bv_data is None:
            return
        if self.is_video_play:
            self.widget_config("btn_play_video", "text", "▶")
            self.is_video_play = False
            return
        if self.objs["scale_get_idx"].get() >= self.bv_data.get_frame_cnt() - 1:
           self.objs["scale_get_idx"].set(0)
        self.is_video_play = True
        self.widget_config("btn_play_video", "text", "■")
        self._play_video()

    def get_analysis_tool(self):
        
        tool_idx = "Joint Angle"
        
        if "combobox_analysis_tool" in self.objs:
            tool_idx = self.objs["combobox_analysis_tool"].get()

        return self.analysis_tool_dict[tool_idx]

    def _play_video(self):

        idx = self.objs["scale_get_idx"].get() + 1

        if not self.is_video_play:
            return
        
        self.objs["scale_get_idx"].set(idx)
        self._show_img_at(idx)
        self.master.after(int(1000 / self.bv_data.raw_video_fps), self._play_video)

    def pause_video(self, event):
        self.is_video_play = False
        self.widget_config("btn_play_video", "text", "▶")
        self.show_img()

    def show_img(self, event=None):
        if self.bv_data is None:
            return
        idx = self.objs["scale_get_idx"].get()
        self._show_img_at(idx)

    def _show_img_at(self, idx):

        #img = self.pose_frame_maker.get_img_at(idx)
        img = self.video_frame_maker.get_img_at(idx)

        if img is None:
            self.is_video_play = False
            self.widget_config("btn_play_video", "text", "▶")
            return

        img_width = img.shape[1]
        img_height = img.shape[0]
        
        img = Image.fromarray(img)
        
        label_width = self.objs["label_pose_img"].winfo_width()
        label_height = self.objs["label_pose_img"].winfo_height()   

        width_ratio = label_width / img_width
        height_ratio = label_height / img_height


        if width_ratio < height_ratio:
            ratio = width_ratio
        else:
            ratio = height_ratio

        img = img.resize((int(img_width * ratio), int(img_height * ratio)))
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.objs["label_pose_img"].config(image=imgtk)
        self.objs["label_pose_img"].image = imgtk # 이미지 객체에 대한 참조를 유지 (가비지 컬렉션 방지)

        return True

    def _update_output(self, message):
        print(message)
        
    def set_bv_data(self, bv_data:ProcessedData):

        self.bv_data = bv_data
        self.df = self.get_analysis_tool().calc(self.bv_data)
        self.set_video_frame_maker()
            
        self.widget_config("scale_get_idx", "to",
                           self.bv_data.get_frame_cnt())
        
        self.objs["scale_get_idx"].set(0)
        self.show_img()

    def process_video(self):
        ProcessApp(self.master, self.set_bv_data)
    
    def open_settings(self):
        ProcessSetting(self.master)

    def open_graph_settings(self):
        GraphSetting(self.master, self.visualize_parameter, 
                     self.get_analysis_tool(), self._graph_settings_changed)
    
    def _graph_settings_changed(self):
        self.set_video_frame_maker()
        self.show_img()
        pass
        
    def show_3d_video(self):
        if self.all_frames_3d_landmark is None:
            self._update_output("몬가 엄슴")
            return
        d3d.show_3d_video(self.all_frames_3d_landmarks, 24)
    
    def load_data(self):
        
        filename = filedialog.askopenfilename(
            title="분석 파일 선택",
            filetypes=(
                ("분석 파일", "*.bv"),
                ("모든 파일", "*.*")
            )
        )

        if not filename:
            return

        data = ProcessedData()
        data.load(filename)

        self.set_bv_data(data)

    def save_data(self):
        if (self.bv_data is None):
            return
        
        filename = filedialog.asksaveasfilename(
            title="저장 위치 선택",
            defaultextension=".bv",
            filetypes=(
                ("분석 파일", "*.bv"),
                ("모든 파일", "*.*")
            ))
        
        if not filename:
            return
        
        self.bv_data.save(filename)
    
    def save_csv(self):
        if (self.df is None):
            return
        
        filename = filedialog.asksaveasfilename(
            title="저장 위치 선택",
            defaultextension=".csv",
            filetypes=(
                ("분석 파일", "*.csv"),
                ("모든 파일", "*.*")
            ))
        
        if not filename:
            return
        
        self.df.to_csv(filename)

    def save_video(self):
        if (self.bv_data is None):
            return
        
        filename = filedialog.asksaveasfilename(
            title="저장 위치 선택",
            defaultextension=".mp4",
            filetypes=(("비디오 파일", "*.mp4"), 
                       ("all files", "*.*")))
        
        if not filename:
            return
        
        self.video_maker.save_video(
            filename,
            self.video_frame_maker,
            self.bv_data.raw_video_fps)