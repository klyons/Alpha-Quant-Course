import sys, pdb, os
import pdb
import warnings
import MetaTrader5 as mt5
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
warnings.filterwarnings("ignore")

# SAVE WEIGHTS
def run(symbol='SPY', timespan='M', multiplier=10, instrument='Equities', opt_params = None,train_length=200_000):
    save = False
    name = f"TreePcaQuantile_{symbol}_{multiplier}{timespan}"
    
    cwd = os.getcwd()
    relative_path = f"quantreo/Data/{instrument}/{multiplier}{timespan}/{symbol}_{multiplier}{timespan}.parquet"
    file_path = os.path.join(cwd, relative_path)
    file_path = os.path.normpath(file_path)
    #instantiate data classes
    DataObj = DataHandler()
    TimeCorrection = TimeframeAnalyzer()
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        # df = df.head(200000)
        if 'high_time' not in df.columns or 'low_time' not in df.columns:
            TimeCorrection.high_low_equities(f'{multiplier}{timespan}')
            df = TimeCorrection.get_data()
            df.to_parquet(file_path)
            #print("Columns 'high_time' or 'low_time' are present in the dataframe.")
    else:       
        if instrument=='Equities':            
            DataObj.get_equity(symbol = symbol, multiplier=multiplier, timespan=timespan)
            TimeCorrection.high_low_equities(f'{multiplier}{timespan}')
            df = TimeCorrection.get_data()
            df.to_parquet(file_path)
        if instrument == 'Currencies':
            DataObj.get_currency(symbol = symbol, timeframe=mt5.TIMEFRAME_M5) # mt5.TIMEFRAME_H1 ect
            TimeCorrection.high_low_currencies(f'{multiplier}{timespan}')
        df = pd.read_parquet(file_path)
    #these parameters are measured in percents so they can stay as is .005 is 0.5%

    params_range = {
        "tp": [0.0005 + i*0.0001 for i in range(5)],
        "sl": [-0.0005 - i*0.0001 for i in range(5)],
    }
  
    params_fixed = {
        "look_ahead_period": 10,
        "sma_fast": 30,
        "sma_slow":80,
        "rsi":14,
        "atr":5,
        "cost": 0.00001,
        "leverage": 5,
        "list_X": ["SMA_diff", "RSI", "ATR","candle_way", "filling", "amplitude", "previous_ret"],
        "train_mode": True,
        "lags": 5,
    }
    # You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
    WFO = WalkForwardOptimization(df, BinLogRegPipeline, params_fixed, params_range,length_train_set=100_000, randomness=1.00, anchored=False)
    WFO.run_optimization()

    # Extract best parameters
    params = WFO.best_params_smoothed[-1]
    print("BEST PARAMETERS")
    print(params)

    model = params["model"]
    if save:
        dump(model, f"../models/saved/{name}_model.jolib")

    # Show the results
    WFO.display()

if __name__ == "__main__":
    #populate with what you want
    
    symbol = 'SPY'
    instrument = 'Equities'
    # use 'M' for minute 'H' for hour and 'S' for second
    timespan = 'M'
    multiplier = 3
    # symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000
    run(symbol=symbol, instrument=instrument, timespan=timespan, multiplier=multiplier )
    
    '''
    
    symbol = 'EURUSD'
    timespan = 'minute'
    multiplier = 5
    train_length = 10_000
    instrument = 'Currencies' #Currency or Equity
    #this takes as inputs symbol timeframe, length, instrument
    run(symbol, timespan, multiplier=multiplier, instrument = instrument) #instrument, timeframe, train_length=10_000, 
    '''