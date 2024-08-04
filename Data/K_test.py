"""
Strategy Explanation:
- We use RSI to understand the force of the market and 2 moving averages to understand the trend
- The goal is to trade divergence
- When there is downward trend (SMAs) and a upward force (RSI), we take a buy position and inversely

- We use the ATR to compute our Take-profit & Stop-loss dynamically
"""

from Quantreo.DataPreprocessing import *

class K_test:

    def __init__(self, data, parameters):
        # Set parameters
        self.data = data
        self.fast_sma, self.slow_sma, self.rsi = parameters["fast_sma"], parameters["slow_sma"], parameters["rsi"]
        self.tp, self.sl = None, None
        self.cost = parameters["cost"]
        self.leverage = parameters["leverage"]
        self.get_features()
        self.start_date_backtest = self.data.index[0]

        # Get Entry parameters
        self.buy, self.sell = False, False
        self.open_buy_price, self.open_sell_price = None, None
        self.entry_time, self.exit_time = None, None

        # Get exit parameters
        self.var_buy_high, self.var_sell_high = None, None
        self.var_buy_low, self.var_sell_low = None, None

        self.output_dictionary = parameters.copy()