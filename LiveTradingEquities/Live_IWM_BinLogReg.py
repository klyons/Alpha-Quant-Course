import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import sys, os

import py4j
current_working_dir = os.path.abspath(os.getcwd())

quantreo_path = os.path.join(current_working_dir, 'Quantreo')
# Add the quantreo folder to the Python path
sys.path.append(quantreo_path)

from Strategies import BinLogRegPipeline

import time
from MetaTrader5 import *
from datetime import datetime, timedelta
from Quantreo.LiveTradingSignal import *
import warnings
libs_path = os.path.join(current_working_dir, 'libs')
if libs_path not in sys.path:
    sys.path.append(libs_path)
    
libs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "libs"))
sys.path.append(libs_path)
from libs import livetrading



#quantreo/LiveTrading/Live_IWM_BinLogReg.py
warnings.filterwarnings("ignore")

symbol = "IWM"
lot = 0.01
magic = 16
timeframe = timeframes_mapping["4-hours"]
pct_tp, pct_sl = 0.001, 0.0007 # DONT PUT THE MINUS SYMBOL ON THE SL

def get_hash(input_string=None):
    if not input_string:
        # Get the current date and timestamp
        current_datetime = datetime.now()
        # Convert the current date and timestamp to a string
        input_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the input string encoded to bytes
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash


while True:
    #change the time condition to be the correct time in Pacific time

        # ! YOU NEED TO HAVE THE SYMBOL IN THE MARKET WATCH TO OPEN OR CLOSE A POSITION
        #selected = mt5.symbol_select(symbol)
        #if not selected:
        #    print(f"\nERROR - Failed to select '{symbol}' in MetaTrader 5 with error :", mt5.last_error())

        # Create the signals
        #the inputs into the model are the time periods for the indicatores used... rsi, moving averages ect.
        #buy, sell = li_2023_02_LogRegQuantile(symbol, timeframe[0], 30, 80, 14, 5, 
        #                                 "../models/saved/BinLogreg_IWM_model.jolib")
        import pdb
        pdb.set_trace()
        exchange = LiveTrading()
        df = exchange.get_quote(symbol, lookback_days=10)
        df = exchange.get_time_bars(df, '60T')
        timeframe = df
        relative_path = f"../copernicus/quantreo/models/saved/BinLogReg_ARKK_1H_model.joblib"
        absolute_path = os.path.abspath(relative_path)    
        buy, sell = BinLogRegLive(symbol, timeframe[0], 20, 60, 14, 5, absolute_path)

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
        if not buy and not sell:
            continue
        # Check for open position, if the position is open and model says to do the opposite then close
        # the position and submit order for the new one. 
        status, pos = exchange.get_open_position(symbol)
        flat_wait = False
        if status:
            # position is open and if the signal is to enter another trade, then close this position
            # and submit another order
            if sell and pos.long_quantity > 0:
                exchange.exit_long(pos)
                flat_wait = True
            else if buy and pos.short_quantity > 0:
                exchange.exit_short(pos)
                flat_wait = True
        while (flat_wait):
            status, pos = exchange.get_open_position(symbol)
            if not status:
                flat_wait = False
                break
            print(f"Waiting on {symbol} to be flat")

        # Send trade to the queue
        order = LiveOrder()
        order.symbol = symbol
        order.instruction = str() # BUY or SELL
        order.price = 0 # the current market price
        order.bar_size = 0
        order.stop_loss = 0 # price + (pct_sl * price or price) - (pct_sl * price) for shorts
        order.profit_tgt = 0 # price + (pct_tp * price or price) - (pct_sl * price) for shorts
        order.quantity = 0  # 1 share for now
        order.hash = get_hash()
        order.strategy_name = "Quantreo"
        if sell:
            order.instruction = "SELL"
            exchange.send_order(order)
        if buy:
            order.instruction = "BUY"
            exchange.send_order(order)
        # Generally you run several asset in the same time, so we put sleep to avoid to do again the
        # same computations several times and therefore increase the slippage for other strategies
        time.sleep(1)