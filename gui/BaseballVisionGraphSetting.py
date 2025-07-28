import tkinter as tk
from tkinter import ttk, messagebox

# GraphSetting 클래스 (수정)
class GraphSetting(tk.Toplevel):
    # 🌟🌟🌟 수정: app_selected_data_ref 매개변수 추가 🌟🌟🌟
    def __init__(self, master, app_selected_data_ref, tool, event):
        super().__init__(master)
        self.master = master
        self.title("그래프 설정")
        self.geometry("350x400")
        self.resizable(False, False)

        # 🌟🌟🌟 추가: MainApplication의 리스트를 직접 참조 🌟🌟🌟
        self.app_selected_data_ref = app_selected_data_ref 
        self.tool = tool
        self.apply_event = event

        self.data_vars = {} 
        self.checkboxes = [] 
        
        self._create_widgets()
        self._load_current_settings() # 수정된 로드 방식 사용

        self.transient(master)
        self.grab_set()
        self.focus_set()

        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.is_saved = False

    def _create_widgets(self):
        settings_frame = ttk.LabelFrame(self, text="그래프 표시 데이터 선택", padding=(15, 10))
        settings_frame.pack(padx=10, pady=10, fill="both", expand=True)

        row = 0
        for display_name in self.tool.items():
            var = tk.BooleanVar()
            self.data_vars[display_name] = var 
            
            chk = ttk.Checkbutton(settings_frame, text=display_name, variable=var)
            chk.grid(row=row, column=0, sticky="w", pady=2, padx=5)
            self.checkboxes.append(chk) 
            row += 1
        
        settings_frame.grid_columnconfigure(0, weight=1)

        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=5)

        ttk.Button(buttons_frame, text="저장", command=self._save_settings).pack(side="left", padx=10)
        ttk.Button(buttons_frame, text="취소", command=self._on_closing).pack(side="left", padx=10)

    def _load_current_settings(self):
        """MainApplication의 selected_graph_data_keys 값을 체크박스에 로드합니다."""
        for internal_key, var in self.data_vars.items():
            # 🌟🌟🌟 수정: self.app_selected_data_ref에서 현재 선택값 확인 🌟🌟🌟
            if internal_key in self.app_selected_data_ref: 
                var.set(True)
            else:
                var.set(False)

    def _save_settings(self):
        """UI에서 선택된 데이터를 MainApplication의 멤버 변수에 저장합니다."""
        try:
            selected_data = []
            for internal_key, var in self.data_vars.items():
                if var.get(): 
                    selected_data.append(internal_key)
            
            # 🌟🌟🌟 수정: MainApplication의 멤버 변수 직접 업데이트 🌟🌟🌟
            self.app_selected_data_ref[:] = selected_data # 리스트의 내용을 업데이트 (참조 유지)
            
            messagebox.showinfo("그래프 설정 저장", "그래프 설정이 성공적으로 저장되었습니다.")
            self.is_saved = True
            self.apply_event()
            self.grab_release()
            self.destroy()

        except Exception as e:
            messagebox.showerror("오류", f"그래프 설정 저장 중 오류 발생: {e}")
            
    def _on_closing(self):
        self.grab_release()
        self.destroy()