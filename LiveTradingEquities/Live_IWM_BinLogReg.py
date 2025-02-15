import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import sys, os
current_working_dir = os.path.abspath(os.getcwd())

quantreo_path = os.path.join(current_working_dir, 'Quantreo')
# Add the quantreo folder to the Python path
sys.path.append(quantreo_path)

from Quantreo.Strategies import BinLogRegPipeline
sys.path.insert(0, '..')

import time
from Quantreo.MetaTrader5 import *
from datetime import datetime, timedelta
from Quantreo.LiveTradingSignal import *
import warnings
from libs import data_feed

#quantreo/LiveTrading/Live_IWM_BinLogReg.py
warnings.filterwarnings("ignore")

symbol = "IWM"
lot = 0.01
magic = 16
timeframe = timeframes_mapping["4-hours"]
pct_tp, pct_sl = 0.0064, 0.0047 # DONT PUT THE MINUS SYMBOL ON THE SL
mt5.initialize()


current_account_info = mt5.account_info()
print("------------------------------------------------------------------")
print(f"Login: {mt5.account_info().login} \tserver: {mt5.account_info().server}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(
    f"Balance: {current_account_info.balance} USD, \t Equity: {current_account_info.equity} USD, \t Profit: {current_account_info.profit} USD")
print("------------------------------------------------------------------")

timeframe_condition = get_verification_time(timeframe[1])

while True:
    #change the time condition to be the correct time in Pacific time
    if datetime.now().strftime("%H:%M:%S") in timeframe_condition:
        print(datetime.now().strftime("%H:%M:%S"))

        # ! YOU NEED TO HAVE THE SYMBOL IN THE MARKET WATCH TO OPEN OR CLOSE A POSITION
        selected = mt5.symbol_select(symbol)
        if not selected:
            print(f"\nERROR - Failed to select '{symbol}' in MetaTrader 5 with error :", mt5.last_error())

        # Create the signals
        #the inputs into the model are the time periods for the indicatores used... rsi, moving averages ect.
        #buy, sell = li_2023_02_LogRegQuantile(symbol, timeframe[0], 30, 80, 14, 5, 
        #                                 "../models/saved/BinLogreg_IWM_model.jolib")
        import pdb
        pdb.set_trace()
        data = DataFeed()
        df = data.get_quote(symbol, lookback_days=10)
        df = data.get_time_bars(df, '60T')
        timeframe = df    
        buy, sell = BinLogRegLive(symbol, timeframe[0], 30, 80, 14, 5, 
                                              "../models/saved/BinLogreg_IWM_model.jolib")

        # Import current open positions
        res = resume()
        '''
        # Here we have a tp-sl exit signal, and we can't open two position on the same asset for the same strategy
        if ("symbol" in res.columns) and ("volume" in res.columns):
            if not ((res["symbol"] == symbol) & (res["volume"] == lot)).any():
                # Run the algorithm
                run(symbol, buy, sell, lot, pct_tp=pct_tp, pct_sl=pct_sl, magic=magic)
                # here we can issue a trade for either Buy or sell
        else:
            run(symbol, buy, sell, lot, pct_tp=pct_tp, pct_sl=pct_sl, magic=magic)
        '''
        # Check for open position, if the position is open and model says to do the opposite then close
        # the position and submit order for the new one. 


        # Send trade to the queue

        # Generally you run several asset in the same time, so we put sleep to avoid to do again the
        # same computations several times and therefore increase the slippage for other strategies
        time.sleep(1)