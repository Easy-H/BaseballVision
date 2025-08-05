from .process_setting_app import ProcessSetting

from baseball_vision.processor import Processor
from baseball_vision.pose_detector import MediaPipePoseDetector
from .progress_bar import ProgressBar

from gui.WidgetBuilder import WidgetBuilder

import utils

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

class ProcessApp(tk.Toplevel, WidgetBuilder):
    # 🌟🌟🌟 수정: app_selected_data_ref 매개변수 추가 🌟🌟🌟
    
    def __init__(self, master, event):
        super().__init__(master)
        
        data = utils.yaml.load_yaml(
            "Layout\BaseballVisionProcessorApp.yaml")
        
        self.set_window(self, data)
        self.create_widget(self, data)
        self._bind_widgets()

        self.transient(master)
        self.grab_set()
        self.focus_set()

        toplevel_width = self.winfo_reqwidth()
        toplevel_height = self.winfo_reqheight()
        master_width = master.winfo_width()
        master_height = master.winfo_height()
        master_x = master.winfo_x()
        master_y = master.winfo_y()
        x = master_x + ((master_width - toplevel_width) // 2)
        y = master_y + ((master_height - toplevel_height) // 2)
        self.geometry(f"+{x}+{y}")


        self.video_path_list = []
        
        self.btns = ["btn_select_video",
                     "btn_process_video",
                     "btn_setting"]
        
        self.processor = Processor()
        self.progress_bar = ProgressBar(self.objs["label_progress"],
                                        self.objs["progress_progress"])
        self.event = event

    def _bind_widgets(self):
        self.widget_config("btn_select_video", "command", self.select_video)
        self.widget_config("btn_process_video", "command", self.process_video)
        self.widget_config("btn_setting", "command", self.open_settings)

    def _enable_action_buttons(self):
        self.widgets_config(self.btns, "state", tk.NORMAL)

    def _disable_action_buttons(self):
        self.widgets_config(self.btns, "state", tk.DISABLED)

    def select_video(self):
        
        default_video_path_list = self.video_path_list

        video_path_list = filedialog.askopenfilenames(
            title="동영상 파일 선택",
            filetypes=(
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv"),
                ("모든 파일", "*.*")
            )
        )
        
        if video_path_list:
            self.video_path_list = video_path_list
        else:
            self.video_path_list = default_video_path_list
        
        self.show_selected_video()

    def show_selected_video(self):
        str_print = "Selected Video"

        for s in self.video_path_list:
            str_print += "\n" +  s 

        self.widget_config(
                "label_video_path_list","text",
                str_print)

    def process_video(self):
        self._disable_action_buttons()
        threading.Thread(target=self.process_video_async, daemon=True).start()
    
    def process_video_async(self):

        if not self.video_path_list:
            return
        
        try:
            self.processor.setting(MediaPipePoseDetector(), progress=self.progress_bar)
            bv_data = self.processor.process_video(self.video_path_list)
            self.event(bv_data)
            self.destroy()

        except Exception as e:
            self._enable_action_buttons()
            pass

    def open_settings(self):
        ProcessSetting(self)
        
    def _create_widgets(self):
        pass