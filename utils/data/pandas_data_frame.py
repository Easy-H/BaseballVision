import pandas as pd
from .data_frame import IDataFrame

class PandasDataFrame(IDataFrame):
    def __init__(self, data=None):
        if data is None:
            self.df = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            df = pd.DataFrame(data)
            df = df.infer_objects(copy=False)
            df_interpolated = df.interpolate(method='linear', axis=0)
            self.df = df_interpolated.ffill().bfill()

    def get_row_count(self):
        return self.df.shape[0]
    
    def get_column_list(self):
        return self.df.columns.to_list()
    
    def get_data_column(self, column):
        return self.df[column]
    
    def get_data_row(self, idx):
        return self.df.iloc[idx].to_dict()

    def get_data_row_range(self, ran):
        return PandasDataFrame(self.df.iloc[ran[0]:ran[1]])
    
    def is_empty(self):
        return self.df.empty
    
    def get_index(self):
        return self.df.index
    
    def get_max_min_values(self, columns):
        # 유효한 컬럼들만 필터링
        valid_columns = [col for col in columns if col in self.df.columns]
        
        if not valid_columns:
            return 0, 180  # 기본값 반환

        # 선택된 컬럼들의 데이터를 추출
        selected_data = self.df[valid_columns]
        
        # NaN을 무시하고 전체 데이터프레임에서 최소/최대값 계산
        min_val = selected_data.min().min()
        max_val = selected_data.max().max()

        # 모든 값이 NaN인 경우 기본값으로 처리
        if pd.isna(min_val) or pd.isna(max_val):
            return 0, 0
            
        return min_val, max_val