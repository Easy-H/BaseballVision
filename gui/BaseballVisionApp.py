from baseball_vision.PoseAnalysisProcessor import PoseAnalysisProcessor
from baseball_vision.InteractivePoseVisualizer import InteractivePoseVisualizer
import baseball_vision.AnalysisTool as bvtool
import baseball_vision.draw_3d as d3d
from gui.StdoutRedirector import StdoutRedirector
from gui.VideoProcessorSetting import VideoProcessorSetting
from gui.BaseballVisionGraphSetting import GraphSetting
from gui.App import App
import utils.yaml
import config

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
import cv2 # OpenCV 라이브러리
from PIL import Image, ImageTk # Pillow 라이브러리
import sys # sys 모듈 임포트
import platform

class BaseballVisionApp(App):

    def __init__(self, master):
        self.master = master
        master.overrideredirect(False)

        data = utils.yaml.load_yaml("AppData.yaml")
        self.create_widget(master, data)
        self._bind_method()

        self.video_path = None
        self.cap = None # OpenCV VideoCapture 객체
        self.playing_video = False # 동영상 재생 상태 플래그

        self.default_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.objs["text_output"])
        self.tool = bvtool.PitcherAnalysisTool(["Pelvis", "Body Twist"])
        self.graph_parameter = ["Pelvis", "Body Twist"]
        self.output_filename = None
        self.action_buttons = ["btn_process_video", "btn_setting", "btn_graph_setting",\
                             "btn_open_video_player", "btn_play_3d_video"]
        
        self.set_widget_state(["btn_open_video_player", "btn_play_3d_video"], tk.DISABLED)
        
        print(f"환경:{ platform.architecture() }")
        print("준비되었습니다. '비디오 선택' 버튼을 눌러주세요.\n")

    def _bind_method(self):
        self.widget_config("btn_select_video", "command", self.select_video)
        self.widget_config("btn_process_video", "command", self.process_video)
        self.widget_config("btn_setting", "command", self.open_settings)
        self.widget_config("btn_graph_setting", "command", self.open_graph_settings)
        self.widget_config("btn_open_video_player", "command", self.open_video_in_external_player)
        self.widget_config("btn_play_3d_video", "command", self.show_3d_video)

    def _update_output(self, message):
        """텍스트 출력 영역에 메시지를 추가하고 스크롤을 내립니다."""
        self.objs["text_output"].config(state=tk.NORMAL)
        self.objs["text_output"].insert(tk.END, message + "\n")
        self.objs["text_output"].see(tk.END) # 가장 아래로 스크롤
        self.objs["text_output"].config(state=tk.DISABLED)

    def select_video(self):
        """1. 버튼을 누르면 비디오 선택 화면"""
        self.playing_video = False # 재생 중이었다면 중지
        if self.cap:
            self.cap.release() # 기존 비디오 캡처 객체 해제
            self.cap = None

        default_file_path = self.video_path

        file_path = filedialog.askopenfilenames(
            title="동영상 파일 선택",
            filetypes=(
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv"),
                ("모든 파일", "*.*")
            )
        )
        if file_path:
            self.video_path = file_path
            self.output_filename = None
            self._update_output(f"'{file_path}' 파일이 선택되었습니다.")
            self.objs["label_video_path"].config(text=f"선택된 비디오: {self.video_path}")
            self.objs["label_display"].config(image='') # 화면 초기화
            self.objs["label_display"].image = None
        else:
            self.video_path = default_file_path
            self._update_output("비디오 선택이 취소되었습니다.")
            self.objs["label_video_path"].config(text=f"선택된 비디오: {self.video_path}")
    def process_video(self):
        threading.Thread(target=self.process_video_async, daemon=True).start()
    def process_video_async(self):
        """2. 비디오 선택 화면을 누르면 프로세스 처리, 처리되는 내용 텍스트로 출력"""
        if not self.video_path:
            return
        
        # 여기에 실제 비디오 처리 로직을 구현합니다.
        # 예: 비디오 길이 가져오기, 프레임 수 세기 등
        try:
            processor = PoseAnalysisProcessor(config.OUTPUT_DIR)
            filename, extension = os.path.splitext(self.video_path[0])
            self._update_output("비디오 처리 중... (이 작업은 몇 초 걸릴 수 있습니다)")

            self.output_filename, self.all_frames_3d_landmarks,\
            self.total_frames, self.fps =\
                processor.process_video(self.video_path,
                                        os.path.basename(filename), self.tool)
            
            self.master.after(0, self.show_matplotlib_image)

            self._update_output("비디오 처리 완료! 추가 작업을 위한 버튼이 활성화됩니다.")
            # 3. 비디오 처리가 완료되면 추가 작업을 위한 버튼 활성화
            self.master.after(0, self._enable_action_buttons) # GUI 스레드에서 버튼 활성화
        except Exception as e:
            self._update_output(f"비디오 처리 중 오류 발생: {e}")
            self.master.after(0, self._disable_action_buttons) # 오류 시 비활성화

    def open_settings(self):
        setting_window = VideoProcessorSetting(self.master)
    def open_graph_settings(self):
        graph_settings_window = GraphSetting(self.master, self.graph_parameter,
                                             self.tool, self.show_matplotlib_image)
    def _enable_action_buttons(self):
        """추가 작업 버튼 활성화"""
        self.set_widget_state(self.action_buttons, tk.NORMAL)
    def _disable_action_buttons(self):
        """추가 작업 버튼 비활성화"""
        self.set_widget_state(self.action_buttons, tk.DISABLED)
    def show_3d_video(self):
        if self.all_frames_3d_landmark is None:
            print("몬가 엄슴")
            return
        if self.fps is None:
            print("이게 엄슴")
            return
        d3d.show_3d_video(self.all_frames_3d_landmarks, self.fps)
    def open_video_in_external_player(self):
        if not self.output_filename:
            messagebox.showerror("오류", "먼저 비디오를 선택해주세요.")
            return

        # 앱 내 재생 중이었다면 중지
        self.playing_video = False
        if self.cap:
            self.cap.release()
            self.cap = None

        print(f"'{os.path.basename(self.output_filename)}' 비디오를 외부 플레이어로 엽니다...")
        try:
            if sys.platform == "win32": # Windows 운영체제
                os.startfile(self.output_filename)
            elif sys.platform == "darwin": # macOS 운영체제
                subprocess.run(["open", self.output_filename])
            else: # Linux 및 기타 유닉스 계열 운영체제
                subprocess.run(["xdg-open", self.output_filename])
            print("외부 플레이어 실행 요청 완료.")
        except Exception as e:
            messagebox.showerror("오류", f"외부 플레이어 실행 실패: {e}")
            print(f"외부 플레이어 실행 실패: {e}")

    # 기존 _play_video_thread (앱 내 재생)와 play_video는 그대로 유지하거나 삭제
    # 이 예제에서는 기존 play_video와 _play_video_thread는 제거하고
    # open_external_player_btn 버튼이 그 자리를 대체했습니다.
    # 만약 앱 내 재생과 외부 재생 두 가지 모두 원하시면, 기존 play_video를 다시 추가하고
    # open_external_player_btn을 새로운 버튼으로 두시면 됩니다.
    def show_matplotlib_image(self):
        if self.output_filename is None:
            return
        self.playing_video = False # 동영상 재생 중지
        if self.cap:
            self.cap.release()
            self.cap = None
        # self.open_external_player_btn.config(text="4. 외부 플레이어로 재생") # 버튼 텍스트 원상복구

        print("그래프를 생성중")
        try:
            label_width = self.objs["label_display"].winfo_width()
            label_height = self.objs["label_display"].winfo_height()
            image_np_array = self.tool.create_graph_image(
                self.total_frames, self.total_frames,label_width, label_height, self.graph_parameter)
            
            # numpy 배열을 Pillow 이미지 객체로 변환
            # draw_image()가 RGB를 반환한다면 Image.fromarray(image_np_array, 'RGB')
            # draw_image()가 BGR을 반환한다면 Image.fromarray(cv2.cvtColor(image_np_array, cv2.COLOR_BGR2RGB))
            img = Image.fromarray(image_np_array) # draw_image()가 RGB를 반환한다고 가정

            # 디스플레이 레이블 크기에 맞게 이미지 리사이즈

            if label_width > 0 and label_height > 0:
                img.thumbnail((label_width, label_height), Image.LANCZOS)
            else:
                img.thumbnail((700, 500), Image.LANCZOS) # 초기 윈도우 크기에 따라 임시로 리사이즈

            imgtk = ImageTk.PhotoImage(image=img)
            self.objs["label_display"].config(image=imgtk)
            self.objs["label_display"].image = imgtk # 이미지 객체에 대한 참조를 유지 (가비지 컬렉션 방지)
            print("그래프 출력 완료.")

        except Exception as e:
            print(f"그래프 출력 중 오류 발생: {e}")
            messagebox.showerror("오류", f"그래프 출력 실패: {e}")