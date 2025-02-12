import sys, pdb, os
import pdb
import warnings
import MetaTrader5 as mt5
import numpy as np
# Get the current working directory
current_working_directory = os.getcwd()
# Construct the path to the quantreo folder
quantreo_path = os.path.join(current_working_directory, 'quantreo')
# Add the quantreo folder to the Python path
sys.path.append(quantreo_path)

from Strategies.BinLogRegPipeline import *  # TreePcaQuantile_Pipeline
from Quantreo.Backtest import Backtest
from Quantreo.WalkForwardOptimization import WalkForwardOptimization
from Data.create_databases import DataHandler
from Data.HighLowTime import TimeframeAnalyzer

import warnings
import argparse
warnings.filterwarnings("ignore")

#get data
def get_data(symbol='SPY', timespan='M', multiplier=30, instrument='Equities'):
    cwd = os.getcwd()
    relative_path = f"quantreo/Data/{instrument}/{multiplier}{timespan}/{symbol}_{multiplier}{timespan}.parquet"
    file_path = os.path.join(cwd, relative_path)
    file_path = os.path.normpath(file_path)
    
    # Instantiate data classes
    DataObj = DataHandler()
    TimeCorrection = TimeframeAnalyzer()
    
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        if 'high_time' not in df.columns or 'low_time' not in df.columns:
            TimeCorrection.high_low_equities(f'{multiplier}{timespan}')
            df = TimeCorrection.get_data()
            df.to_parquet(file_path)
            # print("Columns 'high_time' or 'low_time' are present in the dataframe.")
    else:
        if instrument == 'Equities':
            DataObj.get_equity(symbol=symbol, multiplier=multiplier, timespan=timespan)
            TimeCorrection.high_low_equities(f'{multiplier}{timespan}')
            df = TimeCorrection.get_data()
            print(df)
            pdb.set_trace()
            df.to_parquet(file_path)
        elif instrument == 'Currencies':
            DataObj.get_currency(symbol=symbol, timeframe=mt5.TIMEFRAME_M5)  # mt5.TIMEFRAME_H1 etc.
            TimeCorrection.high_low_currencies(f'{multiplier}{timespan}')
        df = pd.read_parquet(file_path)
    return df


def run(symbol='SPY', timespan='M', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000):
    save = False
    name = f"BinLogReg_{symbol}_{multiplier}{timespan}"
    
    #filter times so only inlcude open market hours
    df = get_data(symbol, timespan, multiplier, instrument)
    
    nyse_open = pd.Timestamp('14:00').time()  # 14:30 UTC
    if timespan == 'M' or timespan == 'S':
        nyse_open = pd.Timestamp('14:30').time()  # 14:30 UTC
    nyse_close = pd.Timestamp('21:00').time()  # 21:00 UTC

    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index)

    # Filter data for NYSE open hours
    filtered_df = df.between_time(nyse_open, nyse_close)
    #if timespan == 'M' or timespan == 'S':
    #    df = df.between_time('09:30', '16:00')
    #if timespan == 'H':
    #    df = df.between_time('09:00', '16:00')
    
    params_range = {
        "tp": [0.0008 + i*0.0001 for i in range(4)],
        "sl": [-0.0008 - i*0.0001 for i in range(4)],
    }

    params_fixed = {
        "look_ahead_period": 5,
        "sma_fast": 15,
        "sma_slow":45,
        "rsi":14,
        "atr":5,
        "cost": 0.00002,
        "leverage": 5,
        "list_X": ['SMA_diff', 'RSI', 'ATR','candle_way', 'filling', 'amplitude', 'previous_ret', 'change', 'dist_vwap'],
        "train_mode": True,
        "lags": 0,
    }
    # You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
    WFO = WalkForwardOptimization(df, BinLogRegPipeline, params_fixed, params_range, length_train_set=1_000, randomness=1.00, anchored=False)
    WFO.run_optimization()

    # Extract best parameters
    if WFO.best_params_smoothed:
        params = WFO.best_params_smoothed[-1]
        print("BEST PARAMETERS")
        print(params)
    else:
        print("No best parameters found.")
        return

    model = params["model"]
    if save:
        dump(model, f"../models/saved/{name}_model.jolib")

    # Show the results
    WFO.display()

if __name__ == "__main__":
    #populate with what you want
    parser = argparse.ArgumentParser(description='Run Walk Forward Optimization')
    parser.add_argument('--symbol', type=str, default='IWM', help='Symbol to run the optimization on')
    parser.add_argument('--timespan', type=str, default='H', help='Timespan for the data')
    parser.add_argument('--multiplier', type=int, default=1, help='Multiplier for the timespan')
    parser.add_argument('--instrument', type=str, default='Equities', help='Type of instrument (Equities or Currencies)')
    parser.add_argument('--train_length', type=int, default=5_000, help='Length of the training set')

    args = parser.parse_args()

    symbol = args.symbol
    timespan = args.timespan
    multiplier = args.multiplier
    instrument = args.instrument
    train_length = args.train_length

    run(symbol=symbol, instrument=instrument, timespan=timespan, multiplier=multiplier, train_length=train_length)
    #symbol = 'IWM'
    #instrument = 'Equities'
    # use 'M' for minute 'H' for hour and 'S' for second
    #timespan = 'H'
    #multiplier = 1
    # symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000
    #run(symbol=symbol, instrument=instrument, timespan=timespan, multiplier=multiplier )
    
    '''
    
    symbol = 'EURUSD'
    timespan = 'minute'
    multiplier = 5
    train_length = 10_000
    instrument = 'Currencies' #Currency or Equity
    #this takes as inputs symbol timeframe, length, instrument
    run(symbol, timespan, multiplier=multiplier, instrument = instrument) #instrument, timeframe, train_length=10_000, 
    '''