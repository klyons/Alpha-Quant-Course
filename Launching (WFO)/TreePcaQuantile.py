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

from Strategies.LI_2023_02_TreePcaQuantile_Pipeline import *  # TreePcaQuantile_Pipeline
from Quantreo.Backtest import Backtest
from Quantreo.WalkForwardOptimization import WalkForwardOptimization
from Data.create_databases import DataHandler
from Data.HighLowTime import TimeframeAnalyzer

import warnings
warnings.filterwarnings("ignore")

# added default parameters
def run(symbol='SPY', timespan='M', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000):
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
        df = df.head(500000)
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
    costs = 0.0001
    params_range = {
        "tp": [0.003 + i*0.002 for i in range(3)],
        "sl": [-0.003 - i*0.002 for i in range(3)],
    }   

    params_fixed = {
        "look_ahead_period": 5,  #this parameter sets the 
        "sma_slow": 60,
        "sma_fast": 20,
        "rsi": 21,
        "atr": 15,
        "cost": costs, # 0.0001,
        "leverage": 5,
        "list_X": ["SMA_diff", "RSI", "ATR", "candle_way", "filling", "amplitude", "SPAN_A", "SPAN_B", "BASE", "STO_RSI",
                "STO_RSI_D", "STO_RSI_K", "previous_ret"],
        "train_mode": True,
    }

    # You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
    WFO = WalkForwardOptimization(df, TreePcaQuantile_Pipeline, params_fixed, params_range,length_train_set=10_000, randomness=1.00)
    WFO.run_optimization()

    # Extract best parameters
    params = WFO.best_params_smoothed[-1]
    print("BEST PARAMETERS")
    print(params)

    # Extract the
    model = params["model"]
    sc = params["sc"]
    pca = params["pca"]

    if save:
        dump(model, f"../models/saved/{name}_model.jolib")
        dump(sc, f"../models/saved/{name}_sc.jolib")
        dump(pca, f"../models/saved/{name}_pca.jolib")

    # Show the results
    WFO.display()

if __name__ == "__main__":
    symbol = 'SPY'
    instrument = 'Equities'
    # use 'M' for minute 'H' for hour and 'S' for second
    timespan = 'M'
    multiplier = 3
    # symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000
    run(symbol=symbol, instrument=instrument, timespan=timespan, multiplier=multiplier )

