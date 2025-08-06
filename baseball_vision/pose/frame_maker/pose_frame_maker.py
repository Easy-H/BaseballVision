from utils.frame_maker import IFrameMaker
from ..processed_data import ProcessedData

import pandas as pd

class IPoseFrameMaker(IFrameMaker):
    def set_data(self, data:ProcessedData, df:pd.DataFrame):
        pass
    def set_focus_label(self, labels):
        self.labels = []
        
        for label in labels:
            if label in self.df.columns.to_list():
                self.labels.append(label)

        if len(self.labels) == 0:
            self.labels = self.df.columns.to_list()
