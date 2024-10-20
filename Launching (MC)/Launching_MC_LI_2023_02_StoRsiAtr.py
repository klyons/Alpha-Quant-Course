# importing the sys module
import sys         
# appending the directory of mod.py 
# in the sys.path list
sys.path.append('C:/ws/ALPHA-QUANT-COURSE/') 
from Quantreo.MonteCarlo import *
from quantreo.Strategies.StoRsiAtr import *
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params = {
    "sto_period": 14.,
    "atr_period": 14.,
    "cost": 0.0001,
    "leverage": 5
}

MC = MonteCarlo(df, StoRsiAtr, params, raw_columns=[], discount_calmar_ratio = 252*6)
MC.generate_paths(10, 2000)
MC.display_results()