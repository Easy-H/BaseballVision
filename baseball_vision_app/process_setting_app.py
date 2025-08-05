import tkinter as tk
from tkinter import filedialog, messagebox
from gui.WidgetBuilder import WidgetBuilder

import utils
import config

import os

class ProcessSetting(tk.Toplevel, WidgetBuilder):
    def __init__(self, master):
        super().__init__(master)

        data = utils.yaml.load_yaml("Layout\PoseDetectorSetting.yaml")
        self.set_window(self, data)
        self.create_widget(self, data)

        self.settings_vars = {}
        # 노이즈 표준편차 Entry 위젯들을 저장할 리스트
        self.noise_std_entries = [] # 🌟🌟🌟 추가: 노이즈 필드 위젯 저장용 🌟🌟🌟
        
        self.current_settings = { }

        for key, value in config.MODEL_CONFIG.items():
            self.current_settings[key] = value

        self._bind_widgets()
        self._load_current_settings()

        # 모달처럼 동작하도록 설정 (이전 코드와 동일)
        self.transient(master)
        self.grab_set()
        self.focus_set()

        # 창 닫기 이벤트에 바인딩
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.is_saved = False 

    def _bind_widgets(self):

        self.settings_vars["MODEL_COMPLEXITY"] = tk.IntVar()
        self.settings_vars["MIN_DETECTION_CONFIDENCE"] = tk.DoubleVar()
        self.settings_vars["MIN_TRACKING_CONFIDENCE"] = tk.DoubleVar()
        self.settings_vars["MIN_DRAW_VISIBILITY"] = tk.DoubleVar()
        self.settings_vars["APPLY_KALMAN_FILTER"] = tk.BooleanVar()
        self.settings_vars["PROCESS_NOISE_STD"] = tk.DoubleVar()
        self.settings_vars["MEASUREMENT_NOISE_STD"] = tk.DoubleVar()
        self.settings_vars["OUTPUT_DIR"] = tk.StringVar()

        for key, value in self.settings_vars.items():
            if key == "APPLY_KALMAN_FILTER":
                self.widget_config(key, "variable", value)
                self.widget_config(key, "command", self._toggle_noise_fields)
                continue
            self.widget_config(key, "textvariable", value)
        
        self.noise_std_entries.append(self.objs["PROCESS_NOISE_STD"])
        self.noise_std_entries.append(self.objs["MEASUREMENT_NOISE_STD"])

        self.widget_config("btn_output_dir_setting", "command", self._select_output_dir)
        self.widget_config("btn_save", "command", self._save_settings)
        self.widget_config("btn_cancel", "command", self._on_closing)
        
        self.objs["frame_setting"].grid_columnconfigure(1, weight=1)

    def _load_current_settings(self):
        """현재 설정값을 GUI 위젯에 로드하고, 초기 상태를 업데이트합니다."""
        for key, tk_var in self.settings_vars.items():
            if key in self.current_settings:
                tk_var.set(self.current_settings[key])
        self._toggle_noise_fields() # 🌟🌟🌟 추가: 초기 로드 후 위젯 상태 업데이트 🌟🌟🌟

    def _select_output_dir(self):
        initial_dir = self.settings_vars["OUTPUT_DIR"].get()
        if not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()
        folder_selected = filedialog.askdirectory(initialdir=initial_dir)
        if folder_selected:
            self.settings_vars["OUTPUT_DIR"].set(folder_selected)

    def _save_settings(self):
        try:
            new_settings = self._get_new_setting()
            self._check_valid(new_settings)
            
            for key, value in new_settings.items():
                config.MODEL_CONFIG[key] = value

            messagebox.showinfo("설정 저장", "설정이 성공적으로 저장되었습니다.")
            self.is_saved = True 
            self._on_closing()

        except ValueError as e:
            messagebox.showerror("입력 오류", f"유효하지 않은 입력입니다: {e}")
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 중 오류 발생: {e}")

    def _get_new_setting(self):
        
        new_settings = { }
        
        for key, value in self.settings_vars.items():
            new_settings[key] = value.get()

        if not self.settings_vars["APPLY_KALMAN_FILTER"].get():
            new_settings["PROCESS_NOISE_STD"] = config.MODEL_CONFIG["PROCESS_NOISE_STD"]
            new_settings["MEASUREMENT_NOISE_STD"] = config.MODEL_CONFIG["MEASUREMENT_NOISE_STD"]
        
        return new_settings

    def _check_valid(self, new_settings):

        if not (0 <= new_settings["MODEL_COMPLEXITY"] <= 2):
            raise ValueError("모델 복잡도는 0, 1, 2 중 하나여야 합니다.")
        
        need_over_than_zero = ["MIN_DETECTION_CONFIDENCE", 
                               "MIN_TRACKING_CONFIDENCE",
                               "MIN_DRAW_VISIBILITY"]
        
        if new_settings["APPLY_KALMAN_FILTER"]:
            need_over_than_zero.append("PROCESS_NOISE_STD")
            need_over_than_zero.append("MEASUREMENT_NOISE_STD")

        for key in need_over_than_zero:
            if new_settings[key] < 0:
                raise ValueError(f"{key} 값은 음수가 될 수 없습니다.")
        
        if not os.path.isdir(new_settings["OUTPUT_DIR"]):
            try:
                os.makedirs(new_settings["OUTPUT_DIR"], exist_ok=True)
            except OSError:
                raise ValueError(f"출력 디렉토리 '{new_settings['OUTPUT_DIR']}'를 생성할 수 없습니다. 유효한 경로를 입력해주세요.")

            
    def _on_closing(self):
        self.grab_release()
        self.destroy()

    # 🌟🌟🌟 핵심 추가: 노이즈 필드 상태를 토글하는 함수 🌟🌟🌟
    def _toggle_noise_fields(self):
        """APPLY_KALMAN_FILTER 체크박스 상태에 따라 노이즈 필드를 활성화/비활성화합니다."""
        is_kalman_applied = self.settings_vars["APPLY_KALMAN_FILTER"].get()
        new_state = tk.NORMAL if is_kalman_applied else tk.DISABLED
        
        self.widgets_config(self.noise_std_entries, "state", new_state)

        # 🌟🌟🌟 추가: 노이즈 필드가 비활성화될 때 기본값으로 설정 🌟🌟🌟
        if not is_kalman_applied:
            self.settings_vars["PROCESS_NOISE_STD"].set(config.MODEL_CONFIG["PROCESS_NOISE_STD"])
            self.settings_vars["MEASUREMENT_NOISE_STD"].set(config.MODEL_CONFIG["MEASUREMENT_NOISE_STD"])