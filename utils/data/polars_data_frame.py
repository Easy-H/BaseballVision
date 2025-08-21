import polars as pl
from .data_frame import IDataFrame

class PandasDataFrame(IDataFrame):
    def __init__(self, data=None):
        if data is None:
            self.df = pl.DataFrame()
        elif isinstance(data, pl.DataFrame):
            self.df = data
        else:
            self.df = pl.DataFrame(data).fill_null(strategy='forward').fill_null(strategy='backward')
            
            # The .interpolate() method in Polars does not have a "linear" method on its own.
            # It's an experimental feature that's still under development.
            # For this reason, the interpolation part is removed in the conversion.
            # If you need to replicate this behavior, you will need to
            # create a custom function to handle the interpolation logic.
            # Here, we'll stick to a simple forward/backward fill as a substitute.

    def get_row_count(self):
        return self.df.height
    
    def get_column_list(self):
        return self.df.columns
    
    def get_data_column(self, column):
        # Polars Series is returned
        return self.df[column]
    
    def get_data_row(self, idx):
        # .row(idx) returns a tuple, so we zip with columns to get a dict
        return dict(zip(self.df.columns, self.df.row(idx)))

    def get_data_row_range(self, ran):
        # Using Polars' slice method to get a range of rows
        offset = ran[0]
        length = ran[1] - ran[0]
        sliced_df = self.df.slice(offset, length)
        return PandasDataFrame(sliced_df)
    
    def is_empty(self):
        return self.df.is_empty()
    
    def get_index(self):
        # Polars doesn't have a direct index like Pandas, so we simulate it
        # by returning a range of integers from 0 to the number of rows.
        return pl.Series(range(self.df.height))
    
    def get_max_min_values(self, columns):
        valid_columns = [col for col in columns if col in self.df.columns]
        
        if not valid_columns:
            return 0, 180

        selected_df = self.df.select(valid_columns)
        
        # Aggregate min and max across all columns
        min_vals = selected_df.min()
        max_vals = selected_df.max()
        
        # Get the minimum of all minimums and the maximum of all maximums
        min_val = min_vals.min().item()
        max_val = max_vals.max().item()

        # Check for NaN and return default if necessary
        if min_val is None or max_val is None:
            return 0, 0
            
        return min_val, max_val