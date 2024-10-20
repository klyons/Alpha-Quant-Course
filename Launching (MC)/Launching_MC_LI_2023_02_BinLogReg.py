# importing the sys module
import sys         
# appending the directory of mod.py 
# in the sys.path list
sys.path.append('C:/ws/ALPHA-QUANT-COURSE/') 

from Quantreo.MonteCarlo import *
from quantreo.Strategies.BinLogReg import *
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params = {
    "tp": 0.0035,
    "sl": -0.0035,
    "look_ahead_period": 20,
    "sma_fast": 30,
    "sma_slow": 80,
    "rsi": 14,
    "atr": 5,
    "cost": 0.0001,
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "ATR"],
    "train_mode": True,
}

MC = MonteCarlo(df, BinLogReg, params, raw_columns=[], discount_calmar_ratio = 252*6)
MC.generate_paths(500, 2000)
MC.display_results()