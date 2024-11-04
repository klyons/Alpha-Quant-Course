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

from Strategies.FracDiffMeanReversion import *  # TreePcaQuantile_Pipeline
from Quantreo.Backtest import Backtest
from Quantreo.WalkForwardOptimization import WalkForwardOptimization
from Data.create_databases import DataHandler
from Data.HighLowTime import TimeframeAnalyzer

import warnings
warnings.filterwarnings("ignore")

#get data
def get_data(symbol='SPY', timespan='M', multiplier=10, instrument='Equities'):
    cwd = os.getcwd()
    relative_path = f"quantreo/Data/{instrument}/{multiplier}{timespan}/{symbol}_{multiplier}{timespan}.parquet"
    file_path = os.path.join(cwd, relative_path)
    file_path = os.path.normpath(file_path)
    
    # Instantiate data classes
    DataObj = DataHandler()
    TimeCorrection = TimeframeAnalyzer()
    
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        # df = df.head(200000)
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
            df.to_parquet(file_path)
        elif instrument == 'Currencies':
            DataObj.get_currency(symbol=symbol, timeframe=mt5.TIMEFRAME_M5)  # mt5.TIMEFRAME_H1 etc.
            TimeCorrection.high_low_currencies(f'{multiplier}{timespan}')
        df = pd.read_parquet(file_path)
    return df


def run(symbol='SPY', timespan='M', multiplier=10, instrument='Equities', opt_params = None,train_length=200_000):
    save = False
    name = f"fracDiff_{symbol}_{multiplier}{timespan}"
    
    df = get_data(symbol, timespan, multiplier, instrument)

    params_range = {
        "tp": [0.0005 + i*0.0001 for i in range(5)],
        "sl": [-0.0005 - i*0.0001 for i in range(5)],
    }

    params_fixed = {
        "fast_sma": 72,
        "slow_sma": 120,
        "rsi": 25,
        "cost": 0.0001,
        "leverage": 5,
        "bb_SD": 2
    }
    # You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
    WFO = WalkForwardOptimization(df, FracDiffMeanReversion, params_fixed, params_range,length_train_set=train_length, randomness=1.00, anchored=False)
    WFO.run_optimization()

    # Extract best parameters
    params = WFO.best_params_smoothed[-1]
    print("BEST PARAMETERS")
    print(params)

    #model = params["model"]
    #if save:
        #dump(model, f"../models/saved/{name}_model.jolib")

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
    run(symbol=symbol, instrument=instrument, timespan=timespan, multiplier=multiplier, train_length=100_000 )
    
    '''
    
    symbol = 'EURUSD'
    timespan = 'minute'
    multiplier = 5
    train_length = 10_000
    instrument = 'Currencies' #Currency or Equity
    #this takes as inputs symbol timeframe, length, instrument
    run(symbol, timespan, multiplier=multiplier, instrument = instrument) #instrument, timeframe, train_length=10_000, 
    '''