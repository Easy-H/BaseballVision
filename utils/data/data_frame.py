# utils/data/data_frame.py
from abc import ABC, abstractmethod

class IDataFrame(ABC):
    @abstractmethod
    def get_row_count(self):
        pass

    @abstractmethod
    def get_column_list(self):
        pass

    @abstractmethod
    def get_data_column(self, column):
        pass

    @abstractmethod
    def get_data_row(self, idx):
        pass

    @abstractmethod
    def get_data_row_range(self, ran):
        pass

    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def get_index(self):
        pass
    
    @abstractmethod
    def get_max_min_values(self, columns):
        pass