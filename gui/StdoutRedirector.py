import tkinter as tk
from tkinter import scrolledtext
import sys
import threading
import time

class StdoutRedirector(object):
    def __init__(self, text_widget, update_interval_ms=1000, max_lines=50):
        self.text_widget = text_widget
        self.buffer = [] # (string, should_delete_last_line_flag) 튜플 저장
        self.lock = threading.Lock()
        self.update_interval_ms = update_interval_ms
        self.max_lines = max_lines
        self.after_id = None
        self._start_update_timer()

    def write(self, string):
        should_delete_last_line = False
        string_to_write = string

        # \r 문자가 포함되어 있다면, 이는 일반적으로 현재 줄을 덮어쓰라는 의미입니다.
        if '\r' in string:
            # \r을 기준으로 문자열을 분리하고, 마지막 부분(가장 최근에 덮어쓸 내용)만 사용합니다.
            parts = string.split('\r')
            string_to_write = parts[-1]
            should_delete_last_line = True

            # 🌟 개선된 부분: 버퍼에 이미 내용이 있고 마지막 내용이 '\r'로 인한 것이라면 대체 🌟
            with self.lock:
                if self.buffer and self.buffer[-1][1]: # 버퍼에 값이 있고, 이전 값도 '\r'로 인한 것이라면
                    self.buffer.pop() # 이전 덮어쓰기 내용을 버퍼에서 제거
                                      # 이렇게 하면 _process_buffer에서 불필요한 삭제 플래그가 누적되지 않습니다.
                # 참고: 현재 로직은 버퍼에 '\n'이 포함되지 않은 상태로 여러 \r 업데이트가 들어올 때
                # 마지막 \r 업데이트만 실제 덮어쓰기 대상으로 간주합니다.
                # \r이 새로운 줄의 시작을 의미하고 이전 \n이 없었다면, 이 방식이 적절합니다.

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
        