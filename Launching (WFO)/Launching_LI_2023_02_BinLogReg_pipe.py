

import sys
sys.path.insert(0, '..')
from quantreo.Strategies.BinLogRegPipeline import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *
import pdb

import warnings
warnings.filterwarnings("ignore")

# SAVE WEIGHTS
def run_wfo(symbol = 'SPY', timeframe='5M', train_length=10_000, instrument='Equities'):
    pdb.set_trace()
    path = f"../Data/{instrument}/{timeframe}/{symbol}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    save = True
    name = f"BinLogReg_{symbol}_{timeframe}"


    #add aditinoal data here
    #df = pd.read_csv("../Data/FixTimeBars/EURUSD_30M_Admiral.csv", index_col="time", parse_dates=True)

    if instrument == 'Currencies':
        params_range = {
            "tp": [0.005 + i*0.001 for i in range(5)],
            "sl": [-0.005 - i*0.001 for i in range(5)],
        }
    else:
        params_range = {
            "tp": [0.5 + i*0.1 for i in range(5)],
            "sl": [-0.5 - i*0.1 for i in range(5)],
        }

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
    timeframe = '5M'
    train_length = 10_000
    instrument = 'Currencies' #Currency or Equity
    #this takes as inputs symbol timeframe, length, instrument
    run_wfo(symbol, timeframe, train_length, instrument) #instrument, timeframe, train_length=10_000, 
    