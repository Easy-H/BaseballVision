from baseball_vision.graph_drawer import GraphDrawer
from ..processed_data import ProcessedData
import pandas as pd
    
class AnalysisTool:
    def __init__(self, label:list):
        self.df_results = pd.DataFrame()
        self.focus_label = label

    def calc(self, data:ProcessedData):
        pass

    def items(self):
        pass
    
    def check_data_exist(self, data:dict, joint_list):
        for joint in joint_list:
            if joint not in [*data.keys()]:
                return False
        return True