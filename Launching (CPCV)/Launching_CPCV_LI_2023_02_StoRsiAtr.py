# importing the sys module
import sys         
 
# appending the directory of mod.py 
# in the sys.path list
sys.path.append('C:/ws/ALPHA-QUANT-COURSE/') 

from quantreo.Strategies.RsiSma import *
from Quantreo.CombinatorialPurgedCV import *
from quantreo.Strategies.StoRsiAtr import *
from Quantreo.CombinatorialPurgedCV import *


df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

params_range = {
    "sto_period": [7.,14.],
    "atr_period": [7.,14.]
}

params_fixed = {
    "cost": 0.0001,
    "leverage": 5
}

CPCV = CombinatorialPurgedCV(data=df, TradingStrategy=StoRsiAtr, fixed_parameters=params_fixed,
                             parameters_range=params_range, N=10, k=2, purge_pct=0.10)
CPCV.get_index_samples()
CPCV.get_sub_samples()
CPCV.run_optimization()
CPCV.get_pbo()
CPCV.display_all_graph()