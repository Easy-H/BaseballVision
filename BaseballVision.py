import tkinter as tk
from gui.LoadingGUI import LoadingGUI

if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.update_idletasks() # Tkinter가 변경 사항을 즉시 적용하도록 합니다.
    root.attributes("-topmost", False)
    root.configure(background='gray14')
    from baseball_vision_app.baseball_vision_app import BaseballVisionApp
    app = BaseballVisionApp(root)
    root.mainloop()