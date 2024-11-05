import sys
sys.path.insert(0, '..')
from quantreo.Strategies.TreePcaQuantilePipeline import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *
import pdb
import warnings
warnings.filterwarnings("ignore")



# SAVE WEIGHTS
def run_wfo(df = None):
    save = False
    name = "LI_2023_02_TreePcaQuantile_EURUSD"

    if df is None:
        df = pd.read_parquet("../Data/Equities/3M/SPY_3M.parquet") #, index_col="time", parse_dates=True
    pdb.set_trace()

    #this is for currencies
    params_range = {
        "tp": [0.002 + i*0.001 for i in range(3)],
        "sl": [-0.002 - i*0.001 for i in range(3)],
    }

    #params_range = {
    #    "tp": [0.25 + i*0.05 for i in range(3)],
    #    "sl": [-0.2 - i*0.05 for i in range(3)],
    #}


    params_fixed = {
        "look_ahead_period": 10,
        "sma_slow": 30,
        "sma_fast": 10,
        "rsi": 21,
        "atr": 15,
        "cost": 0.01, #0.0001,
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
    df = pd.read_parquet("../Data/Equities/3M/SPY_3M.parquet")
    run_wfo(df) 
    
