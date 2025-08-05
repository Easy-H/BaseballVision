from ..GraphDrawer import GraphDrawer
from utils.frame_maker import IFrameMaker

import pandas as pd

class GraphFrameMaker(IFrameMaker):
    def __init__(self, df:pd.DataFrame=None, graph_size:int = 0):
        self.graph_drawer = GraphDrawer()
        self.set_graph(df, graph_size)

    def set_graph(self, df:pd.DataFrame, graph_size):
        if df is None:
            return
        self.df = df
        self.graph_size = graph_size
        self._set_graph_size()
    
    def set_focus_label(self, labels):
        self.labels = labels

    def get_size(self):
        return self.graph_size
    
    def _set_graph_size(self):
        w, h = self.get_size()
        self.graph_drawer.setting(
            df=self.df,
            width=w,
            height=h)
        
    def get_img_at(self, idx:int):
        if idx >= self.df.shape[0] - 1:
            return None
        
        graph_img = self.graph_drawer.create_graph_image(idx, self.labels)

        return graph_img