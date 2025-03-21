import pandas as pd
import numpy as np
import ta
from DataPreprocessing_test import *



class TestStrat:
	def __init__(self, data, parameters):
		self.data = data
		self.fast_sma, self.slow_sma, self.rsi = parameters["fast_sma"], parameters["slow_sma"], parameters["rsi"]
		self.tp, self.sl = parameters["tp"], parameters["sl"]
		self.cost = parameters["cost"]
		self.leverage = parameters["leverage"]
		self.get_features()
		self.start_date_backtest = self.data.index[0]

		# get entry parameters
		self.buy, self.sell = False, False
		self.open_buy_price, self.open_sell_price = None, None
		self.entry_time, self.exit_time = None, None

		# get exit parameters
		self.var_buy_high, self.var_sell_high = None, None
		self.var_buy_low, self.var_sell_low = None, None

		self.output_dictionary = parameters.copy()


	def get_features(self):
		self.data = sma(self.data, "close", self.fast_sma)
		self.data = sma(self.data, "close", self.slow_sma)
		self.data = rsi(self.data, "close", self.rsi)

		self.data["signal"] = 0
		self.data["RSI_lagged"] = self.data["RSI"].shift(1)

		condition_1_buy = self.data[f"SMA_{self.fast_sma}"] < self.data[f"SMA_{self.slow_sma}"]
		condition_1_sell = self.data[f"SMA_{self.fast_sma}"] > self.data[f"SMA_{self.slow_sma}"]

		condition_2_buy = self.data[f"RSI"] > self.data["RSI_lagged"]
		condition_2_sell = self.data[f"RSI"] < self.data["RSI_lagged"]

		self.data.loc[condition_1_buy & condition_2_buy, "signal"] = 1
		self.data.loc[condition_1_sell & condition_2_sell, "signal"] = -1


	def get_entry_signal(self, time):
		""" 
		entry signal
		:param time: TimeStamp of the row
		:return: entry signal of the row and the entry time
		"""
		if len(self.data.loc[:time]) < 2:
			return 0, self.entry_time 
		#create entry signal --> -1, 0 1
		entry_signal = 0
		if self.data.loc[:time]["signal"][-2] == 1:
			entry_signal = 1
		elif self.data.loc[:time]["signal"][-2] == -1:
			entry_signal = -1

		if entry_signal == 1 and not self.buy and not self.sell:
			self.buy = True
			self.open_buy_price = self.data.loc[time]["open"]
			self.entry_buy = time

		elif entry_signal == -1 and not self.buy and not self.sell:
			self.sell = True
			self.open_sell_price = self.data.loc[time]["open"]
			self.entry_sell = time

		return entry_signal, self.entry_time

	def get_exit_signal(self, time):
		""" 
		Exit signal
		:param time: TimeStamp of the row
		"""
		"""
        Take-profit & Stop-loss exit signal
        :param time: TimeStamp of the row
        :return: P&L of the position IF we close it

        **ATTENTION**: If you allow your bot to take a buy and a sell position in the same time,
        you need to return 2 values position_return_buy AND position_return_sell (and sum both for each day)
        """
		# Verify if we need to close a position and update the variations IF we are in a buy position
		if self.buy:
			self.var_buy_high = (self.data.loc[time]["high"] - self.open_buy_price) / self.open_buy_price
			self.var_buy_low = (self.data.loc[time]["low"] - self.open_buy_price) / self.open_buy_price

            # Let's check if AT LEAST one of our threshold are touched on this row
			if (self.tp < self.var_buy_high) and (self.var_buy_low < self.sl):

                # Close with a positive P&L if high_time is before low_time
				if self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
					self.buy = False
					self.open_buy_price = None
					position_return_buy = (self.tp - self.cost) * self.leverage
					self.exit_time = time
					return position_return_buy, self.exit_time

                # Close with a negative P&L if low_time is before high_time
				elif self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
					self.buy = False
					self.open_buy_price = None
					position_return_buy = (self.sl - self.cost) * self.leverage
					self.exit_time = time
					return position_return_buy, self.exit_time

				else:
					self.buy = False
					self.open_buy_price = None
					position_return_buy = 0
					self.exit_time = time
					return position_return_buy, self.exit_time

			elif self.tp < self.var_buy_high:
				self.buy = False
				self.open_buy_price = None
				position_return_buy = (self.tp - self.cost) * self.leverage
				self.exit_time = time
				return position_return_buy, self.exit_time

            # Close with a negative P&L if low_time is before high_time
			elif self.var_buy_low < self.sl:
				self.buy = False
				self.open_buy_price = None
				position_return_buy = (self.sl - self.cost) * self.leverage
				self.exit_time = time
				return position_return_buy, self.exit_time

        # Verify if we need to close a position and update the variations IF we are in a sell position
		if self.sell:
			self.var_sell_high = -(self.data.loc[time]["high"] - self.open_sell_price) / self.open_sell_price
			self.var_sell_low = -(self.data.loc[time]["low"] - self.open_sell_price) / self.open_sell_price

            # Let's check if AT LEAST one of our threshold are touched on this row
			if (self.tp < self.var_sell_low) and (self.var_sell_high < self.sl):

                # Close with a positive P&L if high_time is before low_time
				if self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
					self.sell = False
					self.open_sell_price = None
					position_return_sell = (self.tp - self.cost) * self.leverage
					self.exit_time = time
					return position_return_sell, self.exit_time

                # Close with a negative P&L if low_time is before high_time
				elif self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
					self.sell = False
					self.open_sell_price = None
					position_return_sell = (self.sl - self.cost) * self.leverage
					self.exit_time = time
					return position_return_sell, self.exit_time

				else:
					self.sell = False
					self.open_sell_price = None
					position_return_sell = 0
					self.exit_time = time
					return position_return_sell, self.exit_time

            # Close with a positive P&L if high_time is before low_time
			elif self.tp < self.var_sell_low:
				self.sell = False
				self.open_sell_price = None
				position_return_sell = (self.tp - self.cost) * self.leverage
				self.exit_time = time
				return position_return_sell, self.exit_time

            # Close with a negative P&L if low_time is before high_time
			elif self.var_sell_high < self.sl:
				self.sell = False
				self.open_sell_price = None
				position_return_sell = (self.sl - self.cost) * self.leverage
				self.exit_time = time
				return position_return_sell, self.exit_time

		return 0, None


params = {
    "fast_sma": 60,
    "slow_sma": 120,
    "rsi": 21,
    "tp": 0.005,
    "sl": -0.005,
    "cost": 0.0001,
    "leverage": 5,	
}

df = pd.read_csv("../Data/FixTimeBars/AUDUSD_4H_Admiral_READY.csv", index_col="time", parse_dates=True)

ST = TestStrat(df, params)

for time in df.index:
    open_price, entry_time = ST.get_entry_signal(time)
    print("OPEN", open_price, entry_time)
    
    returns, exit_time = ST.get_exit_signal(time)
    print("EXIT", returns, exit_time)