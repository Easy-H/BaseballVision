from utils.frame_maker import IFrameMaker
from utils.data import PandasDataFrame

from ..processed_data import ProcessedData

class IPoseFrameMaker(IFrameMaker):
    def set_data(self, data:ProcessedData, df:PandasDataFrame):
        if data is None: return
        
        self.df = df
        pass

    def set_focus_label(self, labels):
        self.labels = []
        
        for label in labels:
            if label in self.df.get_column_list():
                self.labels.append(label)

        if len(self.labels) == 0:
            self.labels = self.df.get_column_list()
