from baseball_vision.GraphDrawer import GraphDrawer
from baseball_vision.processed_data import ProcessedData
import pandas as pd
    
class AnalysisTool:
    def __init__(self, label:list):
        self.df_results = pd.DataFrame()
        self.focus_label = label
        self.graph_drawer = GraphDrawer()

    def calc(self, data:ProcessedData):

        self.df_results = pd.DataFrame()
        landmarks_3d_list = data.get_landmarks_3d()
        
        for landmarks_3d in landmarks_3d_list:
            if landmarks_3d is None:
                result = { }
            else:
                result = self.calc_joints(landmarks_3d)
            self.df_results = pd.concat([self.df_results, pd.DataFrame([result])],
                                   ignore_index=True)
            
        return self.df_results
        
    def calc_joints(self, joints):
        pass
    
    def save(self, output_name):
        self.df_results.to_csv(output_name + ".csv", index=True)
    
    def show_graph(self, label=[]):
        # [show_graph 유지] 이 메서드는 새로운 Figure를 생성하여 한 번에 보여주는 용도로 사용됩니다.
        remove_idx = []

        for i in range(len(label)):
            if not label[i] in self.df_results.columns.tolist():
                print(label[i] + "is not correct name")
                remove_idx.append(i)

        remove_idx.reverse()

        for i in range(len(remove_idx)):
            del label[remove_idx[i]]
            
        if len(label)==0:
            label = self.df_results.columns.tolist() 
            
        self.graph_drawer.show_graph()

    def drawer_setting(self, width:int, height:int, total_frame_count:int=None,
                      graph_title:str="Joint Angles Over Time", y_label:str = "Degree"):
        if total_frame_count is None:
            total_frame_count = self.df_results.shape[0]
        self.graph_drawer.setting(width, height, self.items(),
                                  total_frame_count,graph_title, y_label)

    def create_graph_image(self, current_frame_idx:int = None, labels:list = None):
        # [수정 사항 8] 그래프 요소들이 초기화되었는지 확인하고 필요하면 초기화합니다.
        
        # [수정 사항 9] labels 파라미터 처리: 동적으로 플로팅할 라벨 결정

        if current_frame_idx is None:
            current_frame_idx = self.df_results.shape[0]

        labels_to_plot = []
        if labels is None: # labels가 None이면 focus_label 사용
            labels_to_plot = self.focus_label 
            if not labels_to_plot: # focus_label도 비어있으면 모든 df_results 컬럼 중 angle_history_labels에 있는 것만 사용
                labels_to_plot = [col for col in self.df_results.columns.tolist() if col in self.items()]
        elif not labels: # labels 리스트가 비어있다면 df_results의 모든 컬럼 중 angle_history_labels에 있는 것만 사용
            labels_to_plot = [col for col in self.df_results.columns.tolist() if col in self.items()]
        else: # labels가 명시적으로 주어진 경우 해당 라벨만 사용
            labels_to_plot = [label for label in labels if label in self.df_results.columns.tolist() and label in self.items()]

        return self.graph_drawer.create_graph_image(
            self.df_results, current_frame_idx, labels_to_plot)
    
    def close_graph(self):
        self.graph_drawer.close_graph()

    def items(self):
        pass


