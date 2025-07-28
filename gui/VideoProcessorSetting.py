import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from gui.AppTopLevel import AppTopLevel

import utils.yaml

# config.py가 없으면 테스트를 위해 임시로 정의 (이전 코드와 동일)
try:
    import config
except ImportError:
    print("config.py 모듈을 찾을 수 없습니다. 임시 전역 변수를 사용합니다.")
    class Config:
        MODEL_COMPLEXITY = 2
        MIN_DETECTION_CONFIDENCE = 0.3
        MIN_TRACKING_CONFIDENCE = 0.3
        MIN_DRAW_VISIBILITY = 0.2
        PROCESS_NOISE_STD = 0.001
        MEASUREMENT_NOISE_STD = 0.05
        APPLY_KALMAN_FILTER = True
        OUTPUT_DIR = 'result_data'
    config = Config()

class VideoProcessorSetting(AppTopLevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master 
        self.title("설정")
        self.geometry("450x500")
        self.resizable(False, False)

        self.settings_vars = {}
        # 노이즈 표준편차 Entry 위젯들을 저장할 리스트
        self.noise_std_entries = [] # 🌟🌟🌟 추가: 노이즈 필드 위젯 저장용 🌟🌟🌟
        
        self.current_settings = {
            "MODEL_COMPLEXITY": config.MODEL_COMPLEXITY,
            "MIN_DETECTION_CONFIDENCE": config.MIN_DETECTION_CONFIDENCE,
            "MIN_TRACKING_CONFIDENCE": config.MIN_TRACKING_CONFIDENCE,
            "MIN_DRAW_VISIBILITY": config.MIN_DRAW_VISIBILITY,
            "PROCESS_NOISE_STD": config.PROCESS_NOISE_STD,
            "MEASUREMENT_NOISE_STD": config.MEASUREMENT_NOISE_STD,
            "APPLY_KALMAN_FILTER": config.APPLY_KALMAN_FILTER,
            "OUTPUT_DIR": config.OUTPUT_DIR
        }

        data = utils.yaml.load_yaml("Setting.yaml")
        self.create_widget(self, data)
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
                self.objs[key].configure(variable=value)
                self.objs[key].configure(command=self._toggle_noise_fields)
                continue
            self.objs[key].configure(textvariable=value)
        
        self.noise_std_entries.append(self.objs["PROCESS_NOISE_STD"])
        self.noise_std_entries.append(self.objs["MEASUREMENT_NOISE_STD"])

        self.objs["btn_output_dir_setting"].configure(command=self._select_output_dir)
        self.objs["btn_save"].configure(command=self._save_settings)
        self.objs["btn_cancel"].configure(command=self._on_closing)
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
            new_settings = {
                "MODEL_COMPLEXITY": self.settings_vars["MODEL_COMPLEXITY"].get(),
                "MIN_DETECTION_CONFIDENCE": self.settings_vars["MIN_DETECTION_CONFIDENCE"].get(),
                "MIN_TRACKING_CONFIDENCE": self.settings_vars["MIN_TRACKING_CONFIDENCE"].get(),
                "MIN_DRAW_VISIBILITY": self.settings_vars["MIN_DRAW_VISIBILITY"].get(),
                # 🌟🌟🌟 수정: 비활성화 상태일 경우 기본값 사용 🌟🌟🌟
                "PROCESS_NOISE_STD": self.settings_vars["PROCESS_NOISE_STD"].get() if self.settings_vars["APPLY_KALMAN_FILTER"].get() else config.PROCESS_NOISE_STD,
                "MEASUREMENT_NOISE_STD": self.settings_vars["MEASUREMENT_NOISE_STD"].get() if self.settings_vars["APPLY_KALMAN_FILTER"].get() else config.MEASUREMENT_NOISE_STD,
                "APPLY_KALMAN_FILTER": self.settings_vars["APPLY_KALMAN_FILTER"].get(),
                "OUTPUT_DIR": self.settings_vars["OUTPUT_DIR"].get()
            }

            if not (0 <= new_settings["MODEL_COMPLEXITY"] <= 2):
                raise ValueError("모델 복잡도는 0, 1, 2 중 하나여야 합니다.")
            
            for key in ["MIN_DETECTION_CONFIDENCE", "MIN_TRACKING_CONFIDENCE", 
                        "MIN_DRAW_VISIBILITY"]: # 🌟🌟🌟 노이즈 필드는 여기서 검사 제외 🌟🌟🌟
                if new_settings[key] < 0:
                    raise ValueError(f"{key} 값은 음수가 될 수 없습니다.")
            
            # 🌟🌟🌟 노이즈 필드에 대한 추가 유효성 검사 (활성화된 경우에만) 🌟🌟🌟
            if new_settings["APPLY_KALMAN_FILTER"]:
                if new_settings["PROCESS_NOISE_STD"] < 0:
                    raise ValueError("프로세스 노이즈 표준편차는 음수가 될 수 없습니다.")
                if new_settings["MEASUREMENT_NOISE_STD"] < 0:
                    raise ValueError("측정 노이즈 표준편차는 음수가 될 수 없습니다.")

            if not os.path.isdir(new_settings["OUTPUT_DIR"]):
                try:
                    os.makedirs(new_settings["OUTPUT_DIR"], exist_ok=True)
                except OSError:
                    raise ValueError(f"출력 디렉토리 '{new_settings['OUTPUT_DIR']}'를 생성할 수 없습니다. 유효한 경로를 입력해주세요.")

            config.MODEL_COMPLEXITY = new_settings["MODEL_COMPLEXITY"]
            config.MIN_DETECTION_CONFIDENCE = new_settings["MIN_DETECTION_CONFIDENCE"]
            config.MIN_TRACKING_CONFIDENCE = new_settings["MIN_TRACKING_CONFIDENCE"]
            config.MIN_DRAW_VISIBILITY = new_settings["MIN_DRAW_VISIBILITY"]
            config.PROCESS_NOISE_STD = new_settings["PROCESS_NOISE_STD"]
            config.MEASUREMENT_NOISE_STD = new_settings["MEASUREMENT_NOISE_STD"]
            config.APPLY_KALMAN_FILTER = new_settings["APPLY_KALMAN_FILTER"]
            config.OUTPUT_DIR = new_settings["OUTPUT_DIR"]

            messagebox.showinfo("설정 저장", "설정이 성공적으로 저장되었습니다.")
            self.is_saved = True 
            self.grab_release()
            self.destroy() 

        except ValueError as e:
            messagebox.showerror("입력 오류", f"유효하지 않은 입력입니다: {e}")
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 중 오류 발생: {e}")
            
    def _on_closing(self):
        self.grab_release()
        self.destroy()

    # 🌟🌟🌟 핵심 추가: 노이즈 필드 상태를 토글하는 함수 🌟🌟🌟
    def _toggle_noise_fields(self):
        """APPLY_KALMAN_FILTER 체크박스 상태에 따라 노이즈 필드를 활성화/비활성화합니다."""
        is_kalman_applied = self.settings_vars["APPLY_KALMAN_FILTER"].get()
        new_state = tk.NORMAL if is_kalman_applied else tk.DISABLED
        
        self.set_widget_state(self.noise_std_entries, new_state)

        # 🌟🌟🌟 추가: 노이즈 필드가 비활성화될 때 기본값으로 설정 🌟🌟🌟
        if not is_kalman_applied:
            self.settings_vars["PROCESS_NOISE_STD"].set(config.PROCESS_NOISE_STD)
            self.settings_vars["MEASUREMENT_NOISE_STD"].set(config.MEASUREMENT_NOISE_STD)