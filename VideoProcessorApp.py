from baseball_vision.PoseAnalysisProcessor import PoseAnalysisProcessor
from baseball_vision.InteractivePoseVisualizer import InteractivePoseVisualizer
import baseball_vision.AnalysisTool as bvtool
import baseball_vision.draw_3d as d3d
import config

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
import threading
import cv2 # OpenCV 라이브러리
from PIL import Image, ImageTk # Pillow 라이브러리
import sys # sys 모듈 임포트

class StdoutRedirector(object):
    def __init__(self, text_widget, update_interval_ms=100, max_lines=10):
        self.text_widget = text_widget
        self.buffer = [] # (string, should_delete_last_line_flag) 튜플 저장
        self.lock = threading.Lock()
        self.update_interval_ms = update_interval_ms
        self.max_lines = max_lines
        self.after_id = None
        self._start_update_timer()

    def write(self, string):
        should_delete_last_line = False
        # \r 문자가 포함되어 있다면, 이는 일반적으로 현재 줄을 덮어쓰라는 의미입니다.
        if '\r' in string:
            # \r을 기준으로 문자열을 분리하고, 마지막 부분(가장 최근에 덮어쓸 내용)만 사용합니다.
            parts = string.split('\r')
            string_to_write = parts[-1]
            should_delete_last_line = True
        else:
            string_to_write = string

        with self.lock:
            self.buffer.append((string_to_write, should_delete_last_line))

    def flush(self):
        pass

    def _start_update_timer(self):
        # 이미 타이머가 실행 중이라면 취소하고 새로 예약
        if self.after_id is not None:
            self.text_widget.after_cancel(self.after_id)
        # Tkinter의 after 메서드를 사용하여 일정 시간 후 _process_buffer 호출
        self.after_id = self.text_widget.after(self.update_interval_ms, self._process_buffer)

    def _process_buffer(self):
        # Tkinter 메인 스레드에서 호출됩니다.
        with self.lock:
            if not self.buffer:
                self._start_update_timer() # 버퍼가 비어있으면 다음 업데이트를 예약
                return

            # 현재 버퍼에 있는 모든 내용을 결합하고, 마지막 줄 삭제 플래그를 확인합니다.
            combined_content = []
            delete_last_line_in_this_batch = False
            for s, delete_flag in self.buffer:
                combined_content.append(s)
                if delete_flag:
                    delete_last_line_in_this_batch = True
            self.buffer.clear()

        self.text_widget.config(state=tk.NORMAL)

        # 🌟 \r 문자가 감지된 경우 마지막 줄 삭제 🌟
        if delete_last_line_in_this_batch:
            try:
                # 현재 텍스트 위젯에 내용이 있는지 확인합니다.
                # 'end-1c'는 마지막 문자 바로 앞을 의미합니다.
                # 'end-1c linestart'는 마지막 줄의 시작 위치를 가져옵니다.
                if self.text_widget.index('end-1c') != '1.0': # 위젯이 비어있지 않다면
                    last_line_start = self.text_widget.index('end-1c linestart')
                    self.text_widget.delete(last_line_start, 'end-1c') # 마지막 줄의 시작부터 끝까지 삭제
            except tk.TclError:
                # 위젯이 비어있거나 인덱싱 오류가 발생할 수 있으므로 예외 처리
                pass

        # 버퍼의 내용을 한 번에 삽입합니다.
        self.text_widget.insert(tk.END, "".join(combined_content))

        # 🌟 최대 줄 수 제한 (이전 구현과 동일) 🌟
        try:
            current_lines = int(self.text_widget.index('end-1c').split('.')[0])
            # 삽입된 새 텍스트에 포함된 개행 문자(\n) 수를 세어 줄 수 변화를 더 정확히 예측합니다.
            # 이 로직은 `\n`이 없는 긴 줄의 경우 줄 수 계산에 오차가 있을 수 있지만,
            # 일반적으로 print()는 \n으로 끝나므로 유효합니다.
            
            # 여기서 중요한 것은, 실제로 위젯에 표시된 줄 수를 기반으로 하는 것입니다.
            # 'end-1c'가 마지막 글자의 인덱스를 가져오므로, 실제 표시 줄 수와 일치합니다.

            if current_lines > self.max_lines:
                # 초과된 줄만큼 정확히 삭제하는 대신,
                # max_lines를 유지하기 위해 필요한 만큼을 삭제합니다.
                # 예를 들어, max_lines를 1000으로 설정했고 1050줄이 있다면, 50줄을 삭제합니다.
                lines_to_keep = self.max_lines
                # 삭제할 시작 인덱스 계산
                # 1.0은 첫 줄, '1.0 + X lines'는 X줄 아래의 시작을 의미
                delete_until_index = f'{current_lines - lines_to_keep + 1}.0'
                self.text_widget.delete('1.0', delete_until_index)

        except tk.TclError:
            # 텍스트 위젯이 비어있거나, 줄 수가 너무 적어 'end-1c' 인덱싱에 실패할 경우 처리
            pass

        self.text_widget.see(tk.END) # 스크롤을 맨 아래로 이동
        self.text_widget.config(state=tk.DISABLED)

        self._start_update_timer() # 다음 업데이트를 예약
        
class VideoProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("BaseballVision")
        master.geometry("800x700") # 전체 창 크기 설정
        master.resizable(True, True) # 창 크기 조절 가능하게 설정

        self.video_path = None
        self.cap = None # OpenCV VideoCapture 객체
        self.playing_video = False # 동영상 재생 상태 플래그
        self.tool = bvtool.PitcherAnalysisTool(["Pelvis", "Body Twist"])
        
        # UI 요소 생성
        self._create_widgets()
        sys.stdout = StdoutRedirector(self.output_text)

    def _create_widgets(self):
        # 1. 컨트롤 프레임 (버튼 및 입력)
        control_frame = tk.Frame(self.master, bd=2, relief="groove", padx=10, pady=10)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        # 비디오 선택 버튼
        self.select_video_btn = tk.Button(control_frame, text="비디오 선택", command=self.select_video)
        self.select_video_btn.pack(side="left", padx=5, pady=5)

        self.process_video_btn = tk.Button(control_frame, text="작업 시작", command=self.process_video)
        self.process_video_btn.pack(side="right", padx=5, pady=5)

        # 비디오 경로 표시 레이블
        self.video_path_label = tk.Label(control_frame, text="선택된 비디오: 없음", wraplength=400)
        self.video_path_label.pack(side="left", padx=10)

        # 2. 프로세스 출력 텍스트 영역
        self.output_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, height=10)
        self.output_text.pack(fill="x", padx=10, pady=5)
        self.output_text.insert(tk.END, "준비되었습니다. '비디오 선택' 버튼을 눌러주세요.\n")
        self.output_text.config(state=tk.DISABLED) # 초기에는 읽기 전용
        
        # 3. 추가 작업 버튼 프레임 (초기 비활성화)
        action_frame = tk.Frame(self.master, bd=2, relief="groove", padx=10, pady=10)
        action_frame.pack(side="top", fill="x", padx=10, pady=10)


        self.open_external_player_btn = tk.Button(action_frame, text="4. 외부 플레이어로 재생",
                                                  command=self.open_video_in_external_player, state=tk.DISABLED)
        self.open_external_player_btn.pack(side="left", padx=5, pady=5)

        # 4. 동영상/이미지 출력 영역 (Label 사용)
        self.display_label = tk.Label(self.master, bg="black")
        self.display_label.pack(fill="both", expand=True, padx=10, pady=10)

    def _update_output(self, message):
        """텍스트 출력 영역에 메시지를 추가하고 스크롤을 내립니다."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END) # 가장 아래로 스크롤
        self.output_text.config(state=tk.DISABLED)

    def select_video(self):
        """1. 버튼을 누르면 비디오 선택 화면"""
        self.playing_video = False # 재생 중이었다면 중지
        if self.cap:
            self.cap.release() # 기존 비디오 캡처 객체 해제
            self.cap = None
        self.display_label.config(image='') # 화면 초기화
        self.display_label.image = None

        file_path = filedialog.askopenfilenames(
            title="동영상 파일 선택",
            filetypes=(
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv"),
                ("모든 파일", "*.*")
            )
        )
        if file_path:
            self.video_path = file_path
            self.video_path_label.config(text=f"선택된 비디오: {file_path}")
            self._update_output(f"'{file_path}' 파일이 선택되었습니다.")
            # 비디오 처리 시작 (별도 스레드에서 실행)
        else:
            self.video_path = None
            self.video_path_label.config(text="선택된 비디오: 없음")
            self._update_output("비디오 선택이 취소되었습니다.")
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
            self.output_filename, self.all_frames_3d_landmarks, self.total_frames = processor.process_video(
                self.video_path, os.path.basename(filename), self.tool)
            
            self.master.after(0, self.show_matplotlib_image)

            self._update_output("비디오 처리 완료! 추가 작업을 위한 버튼이 활성화됩니다.")
            # 3. 비디오 처리가 완료되면 추가 작업을 위한 버튼 활성화
            self.master.after(0, self._enable_action_buttons) # GUI 스레드에서 버튼 활성화
        except Exception as e:
            self._update_output(f"비디오 처리 중 오류 발생: {e}")
            self.master.after(0, self._disable_action_buttons) # 오류 시 비활성화

    def _enable_action_buttons(self):
        """추가 작업 버튼 활성화"""
        self.open_external_player_btn.config(state=tk.NORMAL)

    def _disable_action_buttons(self):
        """추가 작업 버튼 비활성화"""
        self.open_external_player_btn.config(state=tk.DISABLED)
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
        self.playing_video = False # 동영상 재생 중지
        if self.cap:
            self.cap.release()
            self.cap = None
        # self.open_external_player_btn.config(text="4. 외부 플레이어로 재생") # 버튼 텍스트 원상복구

        print("그래프를 생성하고 출력합니다.")
        try:
            label_width = self.display_label.winfo_width()
            label_height = self.display_label.winfo_height()
            image_np_array = self.tool.create_graph_image(self.total_frames, self.total_frames, label_width, label_height, [])
            
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
            self.display_label.config(image=imgtk)
            self.display_label.image = imgtk # 이미지 객체에 대한 참조를 유지 (가비지 컬렉션 방지)
            print("그래프 출력 완료.")

        except Exception as e:
            print(f"그래프 출력 중 오류 발생: {e}")
            messagebox.showerror("오류", f"그래프 출력 실패: {e}")
