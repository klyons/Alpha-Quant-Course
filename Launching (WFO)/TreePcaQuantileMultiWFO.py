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

from Strategies.TreePcaQuantileMulti import * 
from Strategies.TreePcaQuantilePipeline import *  # TreePcaQuantile_Pipeline
from Quantreo.Backtest import Backtest
from Quantreo.WalkForwardOptimizationMulti import *
from Data.create_databases import DataHandler
from Data.HighLowTime import TimeframeAnalyzer

import warnings
warnings.filterwarnings("ignore")

def truncate_to_common_index(self, df, **kwargs):
    # Start with the index of the main dataframe
    common_index = df.index
    
    # Find the common index with each additional dataframe
    for key, additional_df in kwargs.items():
        common_index = common_index.intersection(additional_df.index)
    
    # Truncate the main dataframe to the common index
    df_truncated = df.loc[common_index]
    
    return df_truncated


def load_and_process_data(instrument, symbol, multiplier, timespan):
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

# added default parameters
def run(symbol='SPY', timespan='M', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000):
    save = False
    name = f"TreePcaQuantile_{symbol}_{multiplier}{timespan}"
    
    #this is the dependent variable
    df = load_and_process_data(instrument=instrument, symbol=symbol, multiplier= multiplier, timespan=timespan)
    df = df.tail(50_000)
    #then we load a variety of independent variables
    #df_aapl = load_and_process_data(instrument=instrument, symbol='AAPL', multiplier= multiplier, timespan=timespan)
    #df_ief = load_and_process_data(instrument=instrument, symbol='IEF', multiplier= multiplier, timespan=timespan)
    #df_ief = load_and_process_data(instrument=instrument, symbol='IEF', multiplier= multiplier, timespan=timespan)
    df_iei = load_and_process_data(instrument=instrument, symbol='IEI', multiplier= multiplier, timespan=timespan)
    #df_shy = load_and_process_data(instrument=instrument, symbol='SHY', multiplier= multiplier, timespan=timespan)
    #df_tlt = load_and_process_data(instrument=instrument, symbol='TLT', multiplier= multiplier, timespan=timespan)
    df = truncate_to_common_index(df, df_iei)
    df_iei = truncate_to_common_index(df_iei, df)
    params_range = {
        "tp": [0.00075 + i*0.0001 for i in range(4)],
        "sl": [-0.00075 - i*0.0001 for i in range(4)],
    }   

    params_fixed = {
        "look_ahead_period": 6,  #this parameter sets the dependent variable
        "sma_slow": 30,
        "sma_fast": 10,
        "rsi": 21,
        "atr": 10,
        "cost": 0.00002, 
        "leverage": 5,
        "list_X": ["SMA_diff", "RSI", "ATR", "candle_way", "filling", "amplitude", "SPAN_A", "SPAN_B", "BASE", "STO_RSI",
                "STO_RSI_D", "STO_RSI_K", "previous_ret"],
        "train_mode": True,
        "lags": 5
    }

    # You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
    WFO = WalkForwardOptimizationMulti(df, TreePcaQuantileMulti, params_fixed, params_range, length_train_set=10_000, randomness=1.00, anchored=False, iei=df_iei)
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
    #class specific parameter
    symbol = 'QQQ'
    instrument = 'Equities'
    # use 'M' for minute 'H' for hour and 'S' for second
    timespan = 'M'
    multiplier = 3
    # symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000
    run(symbol=symbol, instrument=instrument, timespan=timespan, multiplier=multiplier )

