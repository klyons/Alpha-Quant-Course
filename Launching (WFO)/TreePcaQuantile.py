import sys
sys.path.insert(0, '..')
from Strategies.LI_2023_02_TreePcaQuantile_Pipeline import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *
import Data.create_databases
import pdb
import warnings
warnings.filterwarnings("ignore")



# SAVE WEIGHTS
def run_wfo(symbol='SPY', timespan='minute', multiplier=10, instrument='Equities', opt_params = None,train_length=10_000):
    save = False
    name = "LI_2023_02_TreePcaQuantile_EURUSD"
    costs = 0.01
    try:
        df = pd.read_parquet(f"../Data/{instrument}/{timespan}/{symbol}_{timespan}.parquet")
    except:
        if instrument=='Equities':            
            Data.create_databases.get_equity(symbol = symbol, multiplier=multiplier, timespan=timespan)
        
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
    timespan = '10M'
    run_wfo(timespan=timespan)

