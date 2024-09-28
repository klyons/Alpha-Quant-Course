import sys, pdb, os
import pdb
import warnings
import MetaTrader5 as mt5

current_working_directory = os.getcwd()
# Construct the path to the quantreo folder
quantreo_path = os.path.join(current_working_directory, 'quantreo')
# Add the quantreo folder to the Python path
sys.path.append(quantreo_path)
# Get the absolute path of the quantreo folder
#quantreo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'quantreo'))
# Add the quantreo folder to the Python path
#sys.path.append(quantreo_path)
from Strategies.LI_2023_02_BinLogReg_Pipeline import *  # TreePcaQuantile_Pipeline
from Quantreo.Backtest import Backtest
from Quantreo.WalkForwardOptimization import WalkForwardOptimization
from Data.create_databases import DataHandler
from Data.HighLowTime import TimeframeAnalyzer

import warnings
warnings.filterwarnings("ignore")

# SAVE WEIGHTS
def run(symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000):
    save = False
    name = f"TreePcaQuantile_{symbol}_{multiplier}{timespan}"
    
    #need to create different mapping for currencies
    time_mapping = {
                    'minute': 'M',
                    'hour': 'H',
                    'second': 'S'
                }
    cwd = os.getcwd()
    
    relative_path = f"quantreo/Data/{instrument}/{multiplier}{time_mapping[timespan]}/{symbol}_{multiplier}{time_mapping[timespan]}.parquet"
    file_path = os.path.join(cwd, relative_path)
    file_path = os.path.normpath(file_path)
    
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        df = df.head(200000)
        pdb.set_trace()
    else:
        DataObg = DataHandler()
        TimeCorrection = TimeframeAnalyzer() 
        if instrument=='Equities':            
            DataObg.get_equity(symbol = symbol, multiplier=multiplier, timespan=timespan)
            if instrument == 'Equities':
                
                TimeCorrection.high_low_equities(str({multiplier})+time_mapping[timespan])
        if instrument == 'Currencies':
            currency_map = {
                '1M': mt5.TIMEFRAME_M1,
                '5M': mt5.TIMEFRAME_M5,
                '10M': mt5.TIMEFRAME_M10,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            DataObg.get_currency(symbol = symbol, timeframe=currency_map[(time_mapping[{timespan}],{multiplier})]) # mt5.TIMEFRAME_H1 ect
            TimeCorrection.high_low_currencies(str({multiplier})+time_mapping[timespan])
    costs = 0.01
    params_range = {
        "tp": [0.20 + i*0.05 for i in range(1)],
        "sl": [-0.20 - i*0.05 for i in range(1)],
    }

    #this is for currencies
    if instrument == 'Currencies':
        params_range = {
            "tp": [0.005 + i*0.002 for i in range(3)], 
            "sl": [-0.005 - i*0.002 for i in range(3)],
        }
        costs = 0.0001    
    params_fixed = {
        "look_ahead_period": 20,
        "sma_fast": 30,
        "sma_slow":80,
        "rsi":14,
        "atr":5,
        "cost": 0.0001,
        "leverage": 5,
        "list_X": ["SMA_diff", "RSI", "ATR"],
        "train_mode": True,
    }

    # You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
    WFO = WalkForwardOptimization(df, BinLogReg_Pipeline, params_fixed, params_range,length_train_set=5_000)
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
    symbol = 'EURUSD'
    timespan = 'minute'
    multiplier = 5
    train_length = 10_000
    instrument = 'Currencies' #Currency or Equity
    #this takes as inputs symbol timeframe, length, instrument
    run(symbol, timespan, multiplier=multiplier, instrument = instrument) #instrument, timeframe, train_length=10_000, 
    