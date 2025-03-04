import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import sys, os, pdb
import datetime as dt

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

warnings.filterwarnings("ignore")

symbol = "ARKK"
strategy_name = "Quantreo"
lot = 0.01
magic = 16
timeframe = timeframes_mapping["4-hours"]
pct_tp, pct_sl = 0.01, 0.008 # DONT PUT THE MINUS SYMBOL ON THE SL

def within_trading_time():
    # Get the current time
    now = dt.datetime.now()
    current_time = now.time()
    # Define the start and end times
    start_time = dt.time(6, 30)
    end_time = dt.time(13, 1)

    # Check if the current time is within the range
    if start_time <= current_time <= end_time:
        return True
    return False

def get_hash(input_string=None):
    if not input_string:
        # Get the current date and timestamp
        current_datetime = datetime.now()
        # Convert the current date and timestamp to a string
        input_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    sha256_hash = hashlib.sha256()
    # Update the hash object with the input string encoded to bytes
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

exchange = livetrading.LiveTrading()
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
        if not within_trading_time():
            print(f'\r"Outside of trading time....sleeping"', end='\r')
            time.sleep(10)
            continue
        print("\n")
        df = exchange.get_quote(symbol, lookback_days=30)
        df = exchange.get_time_bars(df, '60T')

        relative_path = f"../copernicus/quantreo/models/saved/BinLogReg_ARKK_1H_model.joblib"
        absolute_path = os.path.abspath(relative_path)    
        buy, sell = BinLogRegLive(symbol, df, 20, 60, 14, 5, absolute_path)

        # Import current open positions
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
        open_pos, pos = exchange.get_open_position(symbol)
        flat_wait = False
        if open_pos:
            # position is open and if the signal is to enter another trade, then close this position
            # and submit another order
            if sell and pos.long_quantity > 0:
                exchange.exit_position(self, symbol, 'BUY', pos.long_quantity, strategy_name)
                flat_wait = True
            elif buy and pos.short_quantity > 0:
                exchange.exit_position(self, symbol, 'SELL', pos.short_quantity, strategy_name)
                flat_wait = True
            if buy and pos.long_quantity > 0:
                print(f'\r"Active trade....sleeping"', end='\r')
                time.sleep(5)
                continue # already in active trade
            if sell and pos.short_quantity > 0:
                print(f'\r"Active trade....sleeping"', end='\r')
                time.sleep(5)
                continue # already in an active trade
        while (flat_wait):
            open_pos, pos = exchange.get_open_position(symbol)
            if not open_pos:
                flat_wait = False
                break
            print(f"Waiting on {symbol} position to be flat")

        # Send trade to the queue
        print("\n Send new trade")
        quote = exchange.get_single_quote(symbol) # returns a quotes object
        order = livetrading.LiveOrder()
        order.symbol = symbol
        if sell:
            order.instruction = "SELL"
            order.price = quote.ask
            order.profit_tgt = order.price - (pct_sl * order.price)
            order.stop_loss = order.price + (pct_sl * order.price)
        if buy:
            order.instruction = "BUY"
            order.price = quote.bid
            order.profit_tgt = order.price + (pct_sl * order.price)
            order.stop_loss = order.price - (pct_sl * order.price)
        order.quantity = 1  # 1 share for now
        order.hash = get_hash()
        order.strategy_name = strategy_name
        open_pos, pos = exchange.get_open_position(symbol)
        working_order = exchange.get_working_order(symbol)
        if open_pos or working_order:
            print(f"\nPosition/working order for {order.symbol} is already open, refusing to submit this order")
        else:
            exchange.send_order(order)
        time.sleep(5)