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
# Get the absolute path of the quantreo folder
#quantreo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'quantreo'))
# Add the quantreo folder to the Python path
#sys.path.append(quantreo_path)
pdb.set_trace()
from Strategies.LI_2023_02_TreePcaQuantile_Pipeline import *  # TreePcaQuantile_Pipeline
from Quantreo.Backtest import Backtest
from Quantreo.WalkForwardOptimization import WalkForwardOptimization
from Data.create_databases import DataHandler
from Data.HighLowTime import TimeframeAnalyzer

warnings.filterwarnings("ignore")



# SAVE WEIGHTS
def run(symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000):
    save = False
    name = f"TreePcaQuantile_{symbol}_{multiplier}{timespan}"
    costs = 0.01
    #need to create different mapping for currencies
    time_mapping = {
                    'minute': 'M',
                    'hour': 'H',
                    'second': 'S'
                }
    try:
        df = pd.read_parquet(f"../Data/{instrument}/{timespan}/{symbol}_{time_mapping[timespan]}{multiplier}.parquet")
    except:
        DataObg = DataHandler()
        TimeCorrector = TimeframeAnalyzer() 
        if instrument=='Equities':            
            DataObg.get_equity(symbol = symbol, multiplier=multiplier, timespan=timespan)
            if instrument == 'Equities':
                
                TimeCorrector.high_low_equities(str({multiplier})+time_mapping[timespan])
        if instrument == 'Currencies':
            DataObg.get_currency(symbol = symbol, timeframe=mt5.TIMEFRAME_M5) # mt5.TIMEFRAME_H1 ect
            TimeCorrector.high_low_currencies(str({multiplier})+time_mapping[timespan])
            
    # df = pd.read_parquet("../Data/Equities/3M/SHY_3M.parquet") #, index_col="time", parse_dates=True
    pdb.set_trace()

    params_range = {
        "tp": [0.50 + i*0.05 for i in range(1)],
        "sl": [-0.50 - i*0.05 for i in range(1)],
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
        "sma_slow": 120,
        "sma_fast": 30,
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
    timespan = 'minute'
    multiplier=3
    # symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000
    run(symbol=symbol, instrument=instrument, timespan=timespan, multiplier=multiplier )

