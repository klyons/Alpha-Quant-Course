
#C:\ws\Alpha-Quant-Course\Strategies\KL_2023_02_LogRegPCA_Pipeline.py
import sys
sys.path.insert(0, '..')
from Strategies.KL_2023_02_LogRegPCA_Pipeline import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization import *

import warnings
warnings.filterwarnings("ignore")

# SAVE WEIGHTS
save = False
name = "SHY_3M"


#add aditinoal data here
#change symbol
df = pd.read_parquet("../Data/Equities/3M/SHY_3M.parquet") # , index_col="time", parse_dates=True necessary for currencies.  



#step 1: optimize tp and sl
#step 2: optimize other params that fit the model

params_range = {
    "tp": [0.50 + i*0.1 for i in range(5)],
    "sl": [-0.50 - i*0.1 for i in range(5)],
    "atr":[3,4,5]
}

params_fixed = {
    "look_ahead_period": 20,
    "sma_fast": 30,
    "sma_slow":80,
    "rsi":14,
    "atr":5,
    "cost": 0.01, #0.0001 for currencies
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "candle_way", "filling", "amplitude", "ATR", "SPAN_A", "SPAN_B", "BASE", "previous_ret"],
    "train_mode": True,
}

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
WFO = WalkForwardOptimization(df, LogRegPCA_Pipeline, params_fixed, params_range,length_train_set=10_000)
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




#rename after the strategy
# pass in symbol into function call
# create function to run the code
#create main function to run the additional function