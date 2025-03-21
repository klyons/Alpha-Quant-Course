import os
import glob
import re, pdb
import time
from httpx import get
import numpy as np
import pandas as pd
from torch import mul
from tqdm import tqdm
import MetaTrader5 as mt5
from Data.create_databases import DataHandler
import pyarrow.parquet as pq

from converter import get_path
from lib import databank
from lib import utils

class TimeframeAnalyzer:
    def __init__(self):
        self._data = None

    def find_timestamp_extremum(self, df, df_lower_timeframe):
        """
        :param: df_lowest_timeframe
        :return: self._data with three new columns: Low_time (TimeStamp), High_time (TimeStamp), High_first (Boolean)
        """
        df = df.copy()
        if df_lower_timeframe.empty:
            print(f"Empty dataframe for symbol {df.columns.name}")
            return df
        try:
            df = df.loc[df_lower_timeframe.index[0]:]
        except IndexError as e:
            print("error in lower timeframe index")
            pdb.set_trace()
            raise e
        #df = df.loc[df_lower_timeframe.index[0]:]

        # Set new columns
        df["low_time"] = np.nan
        df["high_time"] = np.nan

        # Loop to find out which of the high or low appears first
        for i in tqdm(range(len(df) - 1)):

            # Extract values from the lowest timeframe dataframe
            start = df.iloc[i:i + 1].index[0]
            end = df.iloc[i + 1:i + 2].index[0]
            row_lowest_timeframe = df_lower_timeframe.loc[start:end].iloc[:-1]

            # Extract Timestamp of the max and min over the period (highest timeframe)
            try:
                high = row_lowest_timeframe["high"].idxmax()
                low = row_lowest_timeframe["low"].idxmin()

                df.loc[start, "low_time"] = low
                df.loc[start, "high_time"] = high

            except Exception as e:
                print(e)
                df.loc[start, "low_time"] = None
                df.loc[start, "high_time"] = None

        # Verify the number of row without both TP and SL on same time
        percentage_good_row = len(df.dropna()) / len(df) * 100
        percentage_garbage_row = 100 - percentage_good_row

        # if percentage_garbage_row<95:
        print(f"WARNINGS: Garbage row: {'%.2f' % percentage_garbage_row} %")
        self._data = df.iloc[:-1]
        return df 
        
    def get_data(self):
        return self._data
    
    def get_path(self, timespan, multiplier):
        cwd = os.getcwd()
        folder_path = os.path.join(cwd, r"quantreo\Data\Equities")
        path = os.path.join(folder_path, f"{multiplier}{timespan[0]}")
        return path
    
    def get_symbol_data(self, symbol, timespan, multiplier):
        db = databank.DataBank()
        folder = self.get_path(timespan, multiplier)
        if not os.path.exists(folder):
            os.makedirs(folder)
        df = db.get_trade_data(symbol, timespan, multiplier, start_date=None, save=True, folder=folder, rename=False)
        df.reset_index(inplace=True, drop=True)
        # Convert date_time to pacific time zone
        if not df.empty:
            utils.convert_to_datetime(df, 'date_time', ctime=None, frmat=None, details=False, pacific_time=True)
        df.set_index("date_time", drop=True, inplace=True)
        return df

    def get_paths(self, cwd, timespan, sub_timeframe_map):
        folder_path = os.path.join(cwd, r"quantreo\Data\Equities")
        parent_path = os.path.join(folder_path, timespan)
        child_timespan = sub_timeframe_map.get(timespan, timespan)
        child_path = os.path.join(folder_path, child_timespan)
        return parent_path, child_path
    
    def high_low_equities(self, timespan):
        cwd = os.getcwd()
        sub_timeframe_map = {
            "1D": "4H",
            "4H": "1H",
            "1H": "30M",
            "30M": "10M",
            "10M": "1M",
            "3M": "1M",
            "1M": "10S",
            "30S": "10S",
        }
        timespan_map_reversed = {v: k for k, v in sub_timeframe_map.items()}

        parent_path, child_path = self.get_paths(cwd, timespan, sub_timeframe_map)

        for file in glob.glob(os.path.join(parent_path, "*.parquet")):
            data = os.path.basename(file)
            instrument, timeframe = data.split('_')[0], data.split('_')[1].split('.')[0]
            sub_time = sub_timeframe_map.get(timeframe, "")
            if sub_time:                
                sub_files = glob.glob(os.path.join(child_path, f"{instrument}_{sub_time}.parquet"))
                numeric_values = int(re.findall(r'\d+', sub_time)[0])

                get_symbol_data(self, instrument, timespan=re.sub(r'\d+', '', sub_time), multiplier=numeric_values)
                sub_files = glob.glob(os.path.join(child_path, f"{instrument}_{sub_time}.parquet"))

                if sub_files:
                    sub_file = sub_files[0]
                    schema = pq.read_schema(file)
                    if 'high_time' not in schema.names and 'low_time' not in schema.names:                        
                        high_tf = pd.read_parquet(file)
                        sub_tf = pd.read_parquet(sub_file)
                        df = self.find_timestamp_extremum(high_tf, sub_tf)
                        if not df.empty:
                            df.to_parquet(file)
                            
    def get_high_low(self, instrument, timespan, multiplier, df):
        cwd = os.getcwd()
        sub_timeframe_map = {
            "1day": "4hour",
            "4hour": "1hour",
            "1hour": "30minute",
            "30minute": "10minute",
            "10minute": "1minute",
            "3minute": "1minute",
            "1minute": "10second",
            "30second": "10second",
        }
        #parent_path, child_path = self.get_path(cwd, timespan, sub_timeframe_map)
        # instrument, timeframe = data.split('_')[0], data.split('_')[1].split('.')[0]
        sub_time = sub_timeframe_map.get(f"{multiplier}{timespan.lower()}", "")
        if sub_time:                
            numeric_values = int(re.findall(r'\d+', sub_time)[0])
            timespan = re.sub(r'\d+', '', sub_time)
            sub_df = self.get_symbol_data(symbol=instrument, timespan=timespan, multiplier=numeric_values)                                
            df = self.find_timestamp_extremum(df, sub_df)
        return df

    def high_low_currencies(self, timespan):
        sub_timeframe_map = {
            "1D": mt5.TIMEFRAME_H4,
            "4H": mt5.TIMEFRAME_H1,
            "1H": mt5.TIMEFRAME_M30,
            "30M": mt5.TIMEFRAME_M10,
            "10M": mt5.TIMEFRAME_M1,
            "5M": mt5.TIMEFRAME_M1,
        }
        sub_timeframe_strings = {
            "1D": "4H",
            "4H": "1H",
            "1H": "30M",
            "30M": "10M",
            "10M": "1M",
            "5M": "1M",
        }
        # Ensure the current working directory is correct
        cwd = os.getcwd()
        print("Current working directory:", cwd)
        # Construct the folder path
        folder_path = os.path.join(cwd, "Currencies", timespan)
        print(f"Folder path: {folder_path}")
        # Get a list of all files in the folder
        files = glob.glob(os.path.join(folder_path, '*'))
        # Iterate over all files in the given timeframe
        for file in files:
            data = os.path.basename(file)
            file_instrument = data.split('_')[0]
            timeframe = data.split('_')[1].split('.')[0]
            sub_time = sub_timeframe_strings.get(timeframe, "")        
            if sub_time:
                sub_folder_path = os.path.join(cwd, f"Currencies/{sub_time}")
                sub_files = glob.glob(os.path.join(sub_folder_path, f"{file_instrument}_{sub_time}.parquet"))
                if not sub_files:
                    mt5.initialize()
                    DataObj = DataHandler()
                    DataObj.get_currency(f"{file_instrument}!", timeframe=sub_timeframe_map[timespan])
                    # Re-check for the sub_file after downloading
                    sub_files = glob.glob(os.path.join(sub_folder_path, f"{file_instrument}_{sub_time}.parquet"))

                if sub_files:
                    # If the data in the smaller timeframe exists, run the function that takes both files
                    sub_file = sub_files[0]
                    high_tf = pd.read_parquet(file)
                    sub_tf = pd.read_parquet(sub_file)
                    self.find_timestamp_extremum(high_tf, sub_tf)

# Example usage
analyzer = TimeframeAnalyzer()
# analyzer.high_low_equities("1D")
# analyzer.high_low_currencies("1D")
