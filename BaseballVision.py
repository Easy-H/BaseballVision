import VideoProcessorApp as vpa
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.update_idletasks() # Tkinter가 변경 사항을 즉시 적용하도록 합니다.
    root.attributes("-topmost", False)
    app = vpa.VideoProcessorApp(root)
    root.mainloop()