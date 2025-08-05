import tkinter as tk
from gui.WidgetBuilder import WidgetBuilder

class LoadingGUI(tk.Tk, WidgetBuilder):
    def __init__(self):
        super().__init__()
        
        self.overrideredirect(True)
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        window_width = 300
        window_height = 200

        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")