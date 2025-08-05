from utils.progress_bar.progress_bar import IProgressBar
import tkinter as tk

class ProgressBar(IProgressBar):
    def __init__(self, label, progress_bar):
        self.label = label
        self.value = tk.DoubleVar()
        self.progress_bar = progress_bar
        self.progress_bar["variable"] = self.value

    def display_progress(self, cur_progress, total_progress):
        ret = cur_progress / total_progress
        self.label["text"] = f"{cur_progress} / {total_progress}"
        self.value.set(ret)
        self.progress_bar.update()
