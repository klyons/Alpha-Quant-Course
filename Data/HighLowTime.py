import numpy as np
from tqdm import tqdm
import pandas as pd
import pdb
import create_databases
import sys
import os, re
import glob
import MetaTrader5 as mt5

def find_timestamp_extremum(df, df_lower_timeframe):
    """
    :param: df_lowest_timeframe
    :return: self._data with three new columns: Low_time (TimeStamp), High_time (TimeStamp), High_first (Boolean)
    """
    df = df.copy()
    df = df.loc[df_lower_timeframe.index[0]:]

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

    df = df.iloc[:-1]

    return df

#potential instruments are "equities" "currencies" and 
def high_low_equities(timespan):
    sub_timeframe_map = {
        "1D": "4H",
        "4H": "1H",
        "1H": "30M",
        "30M": "10M",
        "10M": "1M",
        "1M": "10S",
        "30S": "10S",
    }
    
    timespan_map_reversed = {
        "D": "day",
        "H": "hour",
        "M": "minute",
        "S": "second"
    }
    
    # Ensure the current working directory is correct
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    
    # Construct the folder path
    folder_path = os.path.join(cwd, "Equities", timespan)
    print(f"Folder path: {folder_path}")
    
    # Get a list of all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))
    
    # Iterate over all files in the given timeframe
    for file in files:
        data = os.path.basename(file)
        file_instrument = data.split('_')[0]
        timeframe = data.split('_')[1].split('.')[0]
        sub_time = sub_timeframe_map.get(timeframe, "")
        
        if sub_time:
            sub_folder_path = os.path.join(cwd, instrument, sub_time)
            sub_files = glob.glob(os.path.join(sub_folder_path, f"{file_instrument}_{sub_time}.parquet"))
            
            if not sub_files:
                # If the data in the smaller timeframe doesn't exist, download it
                numeric_values = re.findall(r'\d+', sub_time)
                numeric_string = ''.join(numeric_values)
                create_databases.get_equity(file_instrument, multiplier=int(numeric_string), timespan=timespan_map_reversed[sub_time[-1]])
                # Re-check for the sub_file after downloading
                sub_files = glob.glob(os.path.join(sub_folder_path, f"{file_instrument}_{sub_time}.parquet"))
            
            if sub_files:
                # If the data in the smaller timeframe exists, run the function that takes both files
                sub_file = sub_files[0]
                high_tf = pd.read_parquet(file)
                sub_tf = pd.read_parquet(sub_file)
                find_timestamp_extremum(high_tf, sub_tf)
    
    
def high_low_currencies(timespan):
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
            pdb.set_trace()
            if not sub_files:
                # If the data in the smaller timeframe doesn't exist, download it
                #numeric_values = re.findall(r'\d+', sub_time)
                #numeric_string = ''.join(numeric_values)
                mt5.initialize()
                create_databases.get_currency(f"{file_instrument}!", timeframe=sub_timeframe_map[timespan])
                # Re-check for the sub_file after downloading
                sub_files = glob.glob(os.path.join(sub_folder_path, f"{file_instrument}_{sub_time}.parquet"))
            
            if sub_files:
                # If the data in the smaller timeframe exists, run the function that takes both files
                sub_file = sub_files[0]
                high_tf = pd.read_parquet(file)
                sub_tf = pd.read_parquet(sub_file)
                find_timestamp_extremum(high_tf, sub_tf)        
 # download_data(file_instrument, sub_time)



#df_low_tf = pd.read_csv(r"FixTimeBars/EURUSD_1M.csv", index_col="time", parse_dates=True)
#df_high_tf = pd.read_csv(r"FixTimeBars/EURUSD_5M.csv", index_col="time", parse_dates=True)

#df = find_timestamp_extremum(df_high_tf, df_low_tf)

#print(df[["high_time", "low_time"]])
#df.to_csv("FixTimeBars/EURUSD_1H_R.csv")


#high_low_equities("equities","1hour")
if __name__ == '__main__':
    high_low_currencies("5M")