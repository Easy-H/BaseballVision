from utils.data.graph_drawer import GraphDrawer
from utils.frame_maker import IFrameMaker
from utils.data import PandasDataFrame

import numpy as np

class GraphFrameMaker(IFrameMaker):
    def __init__(self, df:PandasDataFrame=None, graph_size:(int) = 0):
        self.graph_drawer = GraphDrawer()
        self.set_graph(df, graph_size)

    def set_graph(self, df:PandasDataFrame, graph_size):
        if df is None:
            return
        self.df = df
        self.graph_size = graph_size
        self._set_graph_size()
    
    def set_focus_label(self, labels):
        self.labels = []
        
        for label in labels:
            if label in self.df.get_column_list():
                self.labels.append(label)

        if len(self.labels) == 0:
            self.labels = self.df.get_column_list()

    def get_size(self):
        return self.graph_size
    
    def _set_graph_size(self):
        w, h = self.get_size()
        self.graph_drawer.setting(
            df=self.df,
            width=w,
            height=h)
        
    def get_img_at(self, idx:int):

        if idx >= self.df.get_row_count() - 1:
            return None
        
        return self.graph_drawer.create_graph_image(idx, self.labels)