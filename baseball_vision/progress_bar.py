from utils.progress_bar.progress_bar import IProgressBar

class ProgressBar(IProgressBar):
    def __init__(self, std_out):
        self.std_out = std_out
        self.progress_interval = None
        pass
    def display_progress(self, cur_process, total_process):
        if self.progress_interval is None:
            self.progress_interval = max(1, total_process // 100)

        if cur_process <= 0: return
        if cur_process != 1\
            and cur_process % self.progress_interval != 0\
            and cur_process != total_process:
            return
        
        progress_percent = (cur_process / total_process) * 100
        
        self.std_out.write(f"\r처리 중: {progress_percent:.2f}% ({cur_process}/{total_process} 프레임)")
        self.std_out.flush()