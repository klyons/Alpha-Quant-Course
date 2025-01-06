import ta
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from fracdiff.sklearn import Fracdiff, FracdiffStat
import pdb


def breakout(df, n = 10, decay_factor=0.9):
    """
    Generate a signal based on close price breaking 
    above or below the high over the past n datapoints, 
    and apply exponential decay to the signal.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' and 'high' columns.
    n (int): Number of datapoints to look back.
    decay_factor (float): Exponential decay factor. Default is 0.9.

    Returns:
    pd.DataFrame: DataFrame with the breakout signal applied.
    """
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Calculate the rolling high and low over the past n datapoints
    rolling_high = df_copy['high'].rolling(window=n).max()
    rolling_low = df_copy['low'].rolling(window=n).min()

    # Generate the signal
    signal = np.where(df_copy['close'] > rolling_high, 1, np.where(df_copy['close'] < rolling_low, -1, 0))

    # Apply exponential decay to the signal
    df_copy[f'breakout_{n}'] = pd.Series(signal).ewm(alpha=1-decay_factor).mean()

    return df_copy

import pandas as pd
import numpy as np

def alpha_01(df, decay_days=2):
    """
    Calculate the alpha for stocks above and below their VWAP.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' and 'vwap' columns.
    decay_days (int): Number of days for exponential decay. Default is 2.

    Returns:
    pd.Series: Alpha values for the stocks.
    """
    df_copy = df.copy()
    # Calculate relative days since max close
    rel_days_since_max = df_copy['close'].rolling(window=30).apply(np.argmax, raw=True)
    rel_days_since_max_rank = rel_days_since_max.rank()

    # Calculate decline percentage
    decline_pct = (df_copy['vwap'] - df_copy['close']) / df_copy['close']

    # Calculate exponential decay
    decay = rel_days_since_max_rank.ewm(span=decay_days).mean()

    # Calculate alpha for stocks above VWAP
    alpha_above = decline_pct / np.minimum(decay, 0.20)

    # Calculate alpha for stocks below VWAP
    alpha_below = -decline_pct / np.minimum(decay, 0.20)

    # Combine the two alphas
    alpha = np.where(df['close'] > df['vwap'], alpha_above, alpha_below)
    df_copy['alpha_01'] = pd.Series(alpha, index=df.index)
    return df_copy


def get_fractional_diff(df, col, d=0.6):
	"""
	Calculates the fractional difference of a given column in a DataFrame.
	Parameters:
	- df (pd.DataFrame): Input DataFrame containing the column for which to calculate fractional difference.
	- col (str): The name of the column in the DataFrame for which to calculate fractional difference.
	- d (float, optional): The fractional difference parameter. Default is 0.5.
	Returns:
	- pd.DataFrame: A new DataFrame with an additional column named 'frac_diff_{d}', where {d} is the provided parameter.
	"""
	if d == 0:
		f = FracdiffStat()
		X = f.fit_transform(df[col])
		d = f.d_[0]

	df_copy = df.copy()
	fracdiff = Fracdiff(d=d)
	df_copy[f"frac_diff"] = fracdiff.fit_transform(df_copy[col].values.reshape(-1, 1)).flatten()

	return df_copy

def weighted_sum(x: np.ndarray, weights: np.ndarray) -> float:
    return np.sum(x * weights[-len(x):])

def construct_momentum_factors(stock_returns: pd.DataFrame, momentum_params: list, winsorize_percentile: float = 0.05) -> pd.DataFrame:
    """
    Constructs multiple momentum factors based on the provided parameters.
    Each momentum factor has its own window, half-life, and lag.
    """
    for param in momentum_params:
        window = param["window"]
        half_life = param["half_life"]
        lag = param["lag"]
        name = param["name"]
        
        weights = np.exp(-np.log(2) / half_life * np.arange(window))[::-1]
        weights /= weights.sum()
        
        shifted_returns_col = stock_returns["returns"].shift(lag)
        momentum_score = shifted_returns_col.rolling(window=window).apply(lambda x: weighted_sum(x.to_numpy(), weights), raw=False).rename(name)
        
        stock_returns = stock_returns.join(momentum_score)
        
        # Winsorization
        q_min = stock_returns[name].quantile(winsorize_percentile)
        q_max = stock_returns[name].quantile(1 - winsorize_percentile)
        stock_returns[name] = stock_returns[name].clip(q_min, q_max)
        
        # Standardization
        stock_returns[name] = (stock_returns[name] - stock_returns[name].mean()) / stock_returns[name].std()
    
    return stock_returns

def sma(df, col, n):
	df[f"SMA_{n}"] = ta.trend.SMAIndicator(df[col],int(n)).sma_indicator()
	return df

def bollinger_bands(df, col, n, d = 2):
    df = df.copy()
    # Calculate the Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=df[col], window=n, window_dev = d)
    df[f"Bollinger_Middle_{d}"] = indicator_bb.bollinger_mavg()
    df[f"Bollinger_Upper_{d}"] = indicator_bb.bollinger_hband()
    df[f"Bollinger_Lower_{d}"] = indicator_bb.bollinger_lband()
    return df

def sma_diff(df, col, n, m):
	df = df.copy()
	df[f"SMA_d_{n}"] = ta.trend.SMAIndicator(df[col], int(n)).sma_indicator()
	df[f"SMA_d_{m}"] = ta.trend.SMAIndicator(df[col], int(m)).sma_indicator()

	df[f"SMA_diff"] = df[f"SMA_d_{n}"] - df[f"SMA_d_{m}"]
	return df


def rsi(df, col, n):
	df = df.copy()
	df[f"RSI"] = ta.momentum.RSIIndicator(df[col],int(n)).rsi()
	return df

def get_volatility(close,span0=20):
	# simple percentage returns
	df0=close.pct_change()
	# 20 days, a month EWM's std as boundary
	df0=df0.ewm(span=span0).std()
	df0.dropna(inplace=True)
	return df0

def get_3_barriers(volatility, prices, plot = False):
	#create a container
	#can be used
	t_final = 25
	upper_lower_multipliers = [2, 2]
	barriers = pd.DataFrame(columns=['bars_passed','price', 'vert_barrier','top_barrier', 'bottom_barrier'], index = volatility.index)
	for bar, vol in volatility.items():
		bars_passed = len(volatility.loc[volatility.index[0] : bar])
		#set the vertical barrier 
		if (bars_passed + t_final < len(volatility.index) and t_final != 0):
			vert_barrier = volatility.index[bars_passed + t_final]
		else:
			vert_barrier = np.nan
		#set the top barrier
		if upper_lower_multipliers[0] > 0:
			top_barrier = prices.loc[bar] + prices.loc[bar] * upper_lower_multipliers[0] * vol
		else:
			#set it to NaNs
			top_barrier = pd.Series(index=prices.index)
		#set the bottom barrier
		if upper_lower_multipliers[1] > 0:
			bottom_barrier = prices.loc[bar] - prices.loc[bar] * upper_lower_multipliers[1] * vol
		else: 
			#set it to NaNs
			bottom_barrier = pd.Series(index=prices.index)
		barriers.loc[bar, ['bars_passed', 'price', 'vert_barrier','top_barrier', 'bottom_barrier']] = bars_passed, prices.loc[bar], vert_barrier, top_barrier, bottom_barrier
	if plot == True:
		plt.plot(barriers.out,'bo')
		plot.show()
	return barriers

def get_labels(barriers):
	labels = []
	size = []  # percent gained or lost
	for i in range(len(barriers.index)):
		start = barriers.index[i]
		end = barriers.vert_barrier[i]
		if pd.notna(end):
			# assign the initial and final price
			price_initial = barriers.price[start]
			price_final = barriers.price[end]
			# assign the top and bottom barriers
			top_barrier = barriers.top_barrier[i]
			bottom_barrier = barriers.bottom_barrier[i]
			# set the profit taking and stop loss conditions
			condition_pt = (barriers.price[start:end] >= top_barrier).any()
			condition_sl = (barriers.price[start:end] <= bottom_barrier).any()
			# assign the labels
			if condition_pt:
				labels.append(1)
			elif condition_sl:
				labels.append(0)
			else:
				labels.append(np.nan)
			size.append((price_final - price_initial) / price_initial)
		else:
			labels.append(np.nan)
			size.append(np.nan)

	# Create a DataFrame with the index and 'Signal' column
	signals_df = pd.DataFrame({'Signal': labels}, index=barriers.index)
	return signals_df

def get_barriers_signal(df, span0=20, t_final=10, upper_lower_multipliers=[2, 2], plot=False):
	"""
	Processes the data by calculating volatility, barriers, and labels.
	Parameters:
	- df (pd.DataFrame): Input DataFrame containing 'close' prices.
	- span0 (int, optional): Span for the EWM volatility calculation. Default is 20.
	- t_final (int, optional): Time period for the vertical barrier. Default is 10.
	- upper_lower_multipliers (list, optional): Multipliers for the upper and lower barriers. Default is [2, 2].
	- plot (bool, optional): Whether to plot the barriers. Default is False.
	Returns:
	- pd.DataFrame: DataFrame with calculated barriers and labels.
	"""
	# Calculate volatility
	volatility = get_volatility(df['close'], span0=span0)
	
	# Calculate barriers
	barriers = get_3_barriers(volatility, df['close'], plot=plot)
	
	# Calculate labels
	labels = get_labels(barriers)
	
	# Merge barriers and labels into the original DataFrame
	# df = df.join(barriers, how='left')
	df = df.join(labels, how='left')
	return df

def garch_prediction(df, col, window = 10):
	df = df.copy()
	# Fit a GARCH(1, 1) model to the 'col' time series
	model = arch_model(df[col], vol='Garch', p=1, q=1)
	model_fit = model.fit(disp='off')
	# Make a one-step ahead prediction
	prediction = model_fit.forecast(start=0)
	df['GARCH'] = prediction.variance
	return df


def atr(df, n):
	df = df.copy()
	df[f"ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], int(n)).average_true_range()
	return df


def sto_rsi(df, col, n):
	df = df.copy()

	StoRsi = ta.momentum.StochRSIIndicator(df[col], int(n))
	df[f"STO_RSI"] = StoRsi.stochrsi() * 100
	df[f"STO_RSI_D"] = StoRsi.stochrsi_d() * 100
	df[f"STO_RSI_K"] = StoRsi.stochrsi_k() * 100
	return df

def kama(df, col, n):
	"""
	Calculates the Kaufman Adaptive Moving Average (KAMA) for a specified column
	in a DataFrame and adds it as a new column named 'kama_{n}'.

	Parameters:
	-----------
	df : pandas.DataFrame
		The DataFrame containing the column for which KAMA is to be calculated.
	col : str
		The name of the column for which KAMA will be calculated.
a		The window period for KAMA calculation.
	
	Returns:
	--------
	df_copy : pandas.DataFrame
		A new DataFrame with the 'kama_{n}' column added.
	"""
	df_copy = df.copy()
	df_copy[f"kama_{n}"] = ta.momentum.KAMAIndicator(df_copy[col], n).kama()
	
	return df_copy

def gap_detection(df, lookback=2):
	"""
	Detects and calculates the bullish and bearish gaps in the given DataFrame.

	Parameters:
	- df (pd.DataFrame): Input DataFrame with columns 'high' and 'low' representing the high and low prices for each period.
	- lookback (int, optional): Number of periods to look back to detect gaps. Default is 2.

	Returns:
	- pd.DataFrame: DataFrame with additional columns:
		* 'Bullish_gap_sup': Upper boundary of the bullish gap.
		* 'Bullish_gap_inf': Lower boundary of the bullish gap.
		* 'Bearish_gap_sup': Upper boundary of the bearish gap.
		* 'Bearish_gap_inf': Lower boundary of the bearish gap.
		* 'Bullish_gap_size': Size of the bullish gap.
		* 'Bearish_gap_size': Size of the bearish gap.

	The function first identifies the bullish and bearish gaps by comparing the current period's high/low prices
	with the high/low prices of the lookback period. It then calculates the size of each gap and forward-fills any
	missing values in the gap boundaries.
	"""
	df_copy = df.copy()
	df_copy["Bullish_gap_sup"] = np.nan
	df_copy["Bullish_gap_inf"] = np.nan

	df_copy["Bearish_gap_sup"] = np.nan
	df_copy["Bearish_gap_inf"] = np.nan

	df_copy["Bullish_gap"] = 0
	df_copy["Bearish_gap"] = 0

	df_copy.loc[df_copy["high"].shift(lookback) < df_copy["low"], "Bullish_gap_sup"] = df_copy["low"]
	df_copy.loc[df_copy["high"].shift(lookback) < df_copy["low"], "Bullish_gap_inf"] = df_copy["high"].shift(lookback)
	df_copy.loc[df_copy["high"].shift(lookback) < df_copy["low"], "Bullish_gap"] = 1

	df_copy.loc[df_copy["high"] < df_copy["low"].shift(lookback), "Bearish_gap_sup"] = df_copy["low"].shift(lookback)
	df_copy.loc[df_copy["high"] < df_copy["low"].shift(lookback), "Bearish_gap_inf"] = df_copy["high"]
	df_copy.loc[df_copy["high"] < df_copy["low"].shift(lookback), "Bearish_gap"] = 1

	df_copy["Bullish_gap_size"] = df_copy["Bullish_gap_sup"] - df_copy["Bullish_gap_inf"]
	df_copy["Bearish_gap_size"] = df_copy["Bearish_gap_sup"] - df_copy["Bearish_gap_inf"]

	# Fill the missing values by the last one
	df_copy[["Bullish_gap_sup", "Bullish_gap_inf",
		"Bearish_gap_sup", "Bearish_gap_inf"]] = df_copy[["Bullish_gap_sup", "Bullish_gap_inf",
													 "Bearish_gap_sup", "Bearish_gap_inf"]].fillna(method="ffill")

	return df_copy

def displacement_detection(df, type_range="standard", strengh=3, period=100):
	"""
	This function calculates and adds a 'displacement' column to a provided DataFrame. Displacement is determined based on
	the 'candle_range' which is calculated differently according to the 'type_range' parameter. Then, it calculates the
	standard deviation of the 'candle_range' over a given period and sets a 'threshold'. If 'candle_range' exceeds this 'threshold',
	a displacement is detected and marked as 1 in the 'displacement' column.

	Parameters:
	df (pd.DataFrame): The DataFrame to add the columns to. This DataFrame should have 'open', 'close', 'high', and 'low' columns.
	type_range (str, optional): Defines how to calculate 'candle_range'. 'standard' calculates it as the absolute difference between
								'close' and 'open', 'extremum' calculates it as the absolute difference between 'high' and 'low'.
								Default is 'standard'.
	strengh (int, optional): The multiplier for the standard deviation to set the 'threshold'. Default is 3.
	period (int, optional): The period to use for calculating the standard deviation. Default is 100.

	Returns:
	pd.DataFrame: The original DataFrame, but with four new columns: 'candle_range', 'MSTD', 'threshold' and 'displacement'.

	Raises:
	ValueError: If an unsupported 'type_range' is provided.
	"""
	df_copy = df.copy()

	# Choose your type_range
	if type_range == "standard":
		df_copy["candle_range"] = np.abs(df_copy["close"] - df_copy["open"])
	elif type_range == "extremum":
		df_copy["candle_range"] = np.abs(df_copy["high"] - df_copy["low"])
	else:
		raise ValueError("Put a right format of type range")

	# Compute the STD of the candle range
	df_copy["MSTD"] = df_copy["candle_range"].rolling(period).std()
	df_copy["threshold"] = df_copy["MSTD"] * strengh

	# Displacement if the candle range is above the threshold
	df_copy["displacement"] = np.nan
	df_copy.loc[df_copy["threshold"] < df_copy["candle_range"], "displacement"] = 1
	df_copy["variation"] = df_copy["close"] - df_copy["open"]

	# Specify the way of the displacement
	df_copy["green_displacement"] = 0
	df_copy["red_displacement"] = 0

	df_copy.loc[(df_copy["displacement"] == 1) & (0 < df_copy["variation"]), "green_displacement"] = 1
	df_copy.loc[(df_copy["displacement"] == 1) & (df_copy["variation"] < 0), "red_displacement"] = 1

	# Shift by one because we only know that we have a displacement at the end of the candle (BE CAREFUL)
	df_copy["green_displacement"] = df_copy["green_displacement"].shift(1)
	df_copy["red_displacement"] = df_copy["red_displacement"].shift(1)

	df_copy["high_displacement"] = np.nan
	df_copy["low_displacement"] = np.nan

	df_copy.loc[df_copy["displacement"] == 1, "high_displacement"] = df_copy["high"]
	df_copy.loc[df_copy["displacement"] == 1, "low_displacement"] = df_copy["low"]

	df_copy["high_displacement"] = df_copy["high_displacement"].fillna(method="ffill")
	df_copy["low_displacement"] = df_copy["low_displacement"].fillna(method="ffill")

	return df_copy

def auto_corr(df, col, n=50, lag=10):
	"""
	Calculates the autocorrelation for a given column in a Pandas DataFrame, using a specified window size and lag.

	Parameters:
	- df (pd.DataFrame): Input DataFrame containing the column for which to compute autocorrelation.
	- col (str): The name of the column in the DataFrame for which to calculate autocorrelation.
	- n (int, optional): The size of the rolling window for calculation. Default is 50.
	- lag (int, optional): The lag step to be used when computing autocorrelation. Default is 10.

	Returns:
	- pd.DataFrame: A new DataFrame with an additional column named 'autocorr_{lag}', where {lag} is the provided lag value. This column contains the autocorrelation values.
	"""
	df_copy = df.copy()
	df_copy[f'autocorr_{lag}'] = df_copy[col].rolling(window=n, min_periods=n, center=False).apply(lambda x: x.autocorr(lag=lag), raw=False)
	return df_copy

def moving_yang_zhang_estimator(df, window_size=30):
	"""
	Calculate Parkinson's volatility estimator based on high and low prices.

	Parameters:
	-----------
	df : pandas.DataFrame
		DataFrame containing 'high' and 'low' columns for each trading period.

	Returns:
	--------
	volatility : float
		Estimated volatility based on Parkinson's method.
	"""
	def yang_zhang_estimator(df):
		N = len(window)
	
		term1 = np.log(window['high'] / window['close']) * np.log(window['high'] / window['open'])
		term2 = np.log(window['low'] / window['close']) * np.log(window['low'] / window['open'])

		sum_squared = np.sum(term1 + term2)
		volatility = np.sqrt(sum_squared / N)

		return volatility
	
	df_copy = df.copy()
	
	# Create an empty series to store mobile volatility
	rolling_volatility = pd.Series(dtype='float64')

	# Browse the DataFrame by window size `window_size` and apply `yang_zhang_estimator`.
	for i in range(window_size, len(df)):
		window = df_copy.loc[df_copy.index[i-window_size]: df_copy.index[i]]
		volatility = yang_zhang_estimator(window)
		rolling_volatility.at[df_copy.index[i]] = volatility

	# Add the mobile volatility series to the original DataFrame
	df_copy['rolling_volatility_yang_zhang'] = rolling_volatility
	
	return df_copy

def rolling_adf(df, col, window_size=30):
	"""
	Calculate the Augmented Dickey-Fuller test statistic on a rolling window.

	Parameters:
	-----------
	df : pandas.DataFrame
		DataFrame containing the column on which to perform the ADF test.
	col : str
		The name of the column on which to perform the ADF test.
	window_size : int
		The size of the rolling window.

	Returns:
	--------
	df_copy : pandas.DataFrame
		A new DataFrame with an additional column containing the rolling ADF test statistic.
	"""

	df_copy = df.copy()
	
	# Create an empty series to store rolling ADF test statistic
	rolling_adf_stat = pd.Series(dtype='float64', index=df_copy.index)

	# Loop through the DataFrame by `window_size` and apply `adfuller`.
	for i in range(window_size, len(df)):
		window = df_copy[col].iloc[i-window_size:i]
		adf_result = adfuller(window)
		adf_stat = adf_result[0]
		rolling_adf_stat.at[df_copy.index[i]] = adf_stat

	# Add the rolling ADF test statistic series to the original DataFrame
	df_copy['rolling_adf_stat'] = rolling_adf_stat
	
	return df_copy

def spread(df):
	"""
	Calculates the spread between the 'high' and 'low' columns of a given DataFrame 
	and adds it as a new column named 'spread'.

	Parameters:
	-----------
	df : pandas.DataFrame
		The DataFrame containing the 'high' and 'low' columns for which the spread is to be calculated.

	Returns:
	--------
	df_copy : pandas.DataFrame
		A new DataFrame with the 'spread' column added.
	"""
	df_copy = df.copy()
	df_copy["spread"] = df_copy["high"] - df_copy["low"]
	
	return df_copy

def kama_market_regime(df, col, n, m):
	"""
	Calculates the Kaufman's Adaptive Moving Average (KAMA) to determine market regime.
	
	Parameters:
	- df (pd.DataFrame): Input DataFrame containing price data or other numeric series.
	- col (str): The column name in the DataFrame to apply KAMA.
	- n (int): The period length for the first KAMA calculation.
	- m (int): The period length for the second KAMA calculation.

	Returns:
	- pd.DataFrame: DataFrame with additional columns "kama_diff" and "kama_trend" indicating the market trend.
	"""
	
	df_copy = df.copy()
	df_copy = kama(df_copy, col, n)
	df_copy = kama(df_copy, col, m)

	df_copy["kama_diff"] = df_copy[f"kama_{m}"] - df_copy[f"kama_{n}"]
	df_copy["kama_trend"] = -1
	df_copy.loc[0<df_copy["kama_diff"], "kama_trend"] = 1
	
	return df_copy

def DC_market_regime(df, threshold):
	"""
	Determines the market regime based on Directional Change (DC) and trend events.
	
	Parameters:
	-----------
	df : pandas.DataFrame
		A DataFrame containing financial data. The DataFrame should contain a 'close' column 
		with the closing prices, and 'high' and 'low' columns for high and low prices.
	
	threshold : float
		The percentage threshold for DC events.
	
	Returns:
	--------
	df_copy : pandas.DataFrame
		A new DataFrame containing the original data and a new column "market_regime", 
		which indicates the market regime at each timestamp. A value of 1 indicates 
		an upward trend, and a value of 0 indicates a downward trend.
		
	"""
	def dc_event(Pt, Pext, threshold):
		"""
		Compute if we have a POTENTIAL DC event
		"""
		var = (Pt - Pext) / Pext

		if threshold <= var:
			dc = 1
		elif var <= -threshold:
			dc = -1
		else:
			dc = 0

		return dc


	def calculate_dc(df, threshold):
		"""
		Compute the start and the end of a DC event
		"""

		# Initialize lists to store DC and OS events
		dc_events_up = []
		dc_events_down = []
		dc_events = []
		os_events = []

		# Initialize the first DC event
		last_dc_price = df["close"][0]
		last_dc_direction = 0  # +1 for up, -1 for down

		# Initialize the current Min & Max for the OS events
		min_price = last_dc_price
		max_price = last_dc_price
		idx_min = 0
		idx_max = 0


		# Iterate over the price list
		for i, current_price in enumerate(df["close"]):

			# Update min & max prices
			try:
				max_price = df["high"].iloc[dc_events[-1][-1]:i].max()
				min_price = df["low"].iloc[dc_events[-1][-1]:i].min()
				idx_min = df["high"].iloc[dc_events[-1][-1]:i].idxmin()
				idx_max = df["low"].iloc[dc_events[-1][-1]:i].idxmax()
			except Exception as e:
				pass
				#print(e, dc_events, i)
				#print("We are computing the first DC")

			# Calculate the price change in percentage
			dc_price_min = dc_event(current_price, min_price, threshold)
			dc_price_max = dc_event(current_price, max_price, threshold)


			# Add the DC event with the right index IF we are in the opposite way
			# Because if we are in the same way, we just increase the OS event size
			if (last_dc_direction!=1) & (dc_price_min==1):
				dc_events_up.append([idx_min, i])
				dc_events.append([idx_min, i])
				last_dc_direction = 1

			elif (last_dc_direction!=-1) & (dc_price_max==-1):
				dc_events_down.append([idx_max, i])
				dc_events.append([idx_max, i])
				last_dc_direction = -1

		return dc_events_up, dc_events_down, dc_events

def dc_event(Pt, Pext, threshold):
	"""
	Compute if we have a POTENTIAL DC event
	"""
	var = (Pt - Pext) / Pext
	
	if threshold <= var:
		dc = 1
	elif var <= -threshold:
		dc = -1
	else:
		dc = 0
		
	return dc


def calculate_dc(df, threshold):
	"""
	Compute the start and the end of a DC event
	"""
	
	# Initialize lists to store DC and OS events
	dc_events_up = []
	dc_events_down = []
	dc_events = []
	os_events = []

	# Initialize the first DC event
	last_dc_price = df["close"][0]
	last_dc_direction = 0  # +1 for up, -1 for down
	
	# Initialize the current Min & Max for the OS events
	min_price = last_dc_price
	max_price = last_dc_price
	idx_min = 0
	idx_max = 0

	
	# Iterate over the price list
	for i, current_price in enumerate(df["close"]):
		
		# Update min & max prices
		try:
			max_price = df["high"].iloc[dc_events[-1][-1]:i].max()
			min_price = df["low"].iloc[dc_events[-1][-1]:i].min()
			idx_min = df["high"].iloc[dc_events[-1][-1]:i].idxmin()
			idx_max = df["low"].iloc[dc_events[-1][-1]:i].idxmax()
		except Exception as e:
			pass
			#print(e, dc_events, i)
			#print("We are computing the first DC")
		
		# Calculate the price change in percentage
		dc_price_min = dc_event(current_price, min_price, threshold)
		dc_price_max = dc_event(current_price, max_price, threshold)
		
		
		# Add the DC event with the right index IF we are in the opposite way
		# Because if we are in the same way, we just increase the OS event size
		if (last_dc_direction!=1) & (dc_price_min==1):
			dc_events_up.append([idx_min, i])
			dc_events.append([idx_min, i])
			last_dc_direction = 1
			
		elif (last_dc_direction!=-1) & (dc_price_max==-1):
			dc_events_down.append([idx_max, i])
			dc_events.append([idx_max, i])
			last_dc_direction = -1
		
	return dc_events_up, dc_events_down, dc_events

def calculate_trend(dc_events_down, dc_events_up, df):
	"""
	Compute the DC + OS period (trend) using the DC event lists
	"""
	
	# Initialize the variables
	trend_events_up = []
	trend_events_down = []
	
	# Verify which event occured first (upward or downward movement)
	
	# If the first event is a downward event
	if dc_events_down[0][0]==0:
		
		# Iterate on the index 
		for i in range(len(dc_events_down)):
			
			# If it is the value before the last one we break the loop
			if i==len(dc_events_down)-1:
				break
				
			# Calculate the start and end for each trend
			trend_events_up.append([dc_events_up[i][1], dc_events_down[i+1][1]])
			trend_events_down.append([dc_events_down[i][1], dc_events_up[i][1]])
	
	# If the first event is a upward event
	elif dc_events_up[0][0]==0:
		
		# Iterate on the index
		for i in range(len(dc_events_up)):
			
			# If it is the value before the last one we break the loop
			if i==len(dc_events_up)-1:
				break
				
			# Calculate the start and end for each trend
			trend_events_up.append([dc_events_down[i][1], dc_events_up[i+1][1]])
			trend_events_down.append([dc_events_up[i][1], dc_events_down[i][1]])

	# Verify the last indexed value for the down ward and the upward trends
	last_up = trend_events_up[-1][1]
	last_down = trend_events_down[-1][1]

	# Find which trend occured last to make it go until now
	if last_down < last_up:
		trend_events_up[-1][1] = len(df)-1
	else:
		trend_events_down[-1][1] = len(df)-1
		
	return trend_events_down, trend_events_up

def get_dc_price(dc_events, df):
	dc_events_prices = []
	for event in dc_events:
		prices = [df["close"].iloc[event[0]], df["close"].iloc[event[1]]]
		dc_events_prices.append(prices)
	return dc_events_prices

def ichimoku(df,n1,n2):
	ICHIMOKU = ta.trend.IchimokuIndicator(df["high"], df["low"], int(n1), int(n2))
	df["SPAN_A"] = ICHIMOKU.ichimoku_a()
	df["SPAN_B"] = ICHIMOKU.ichimoku_b()
	df["BASE"] = ICHIMOKU.ichimoku_base_line()
	df["CONVERSION"] = ICHIMOKU.ichimoku_conversion_line()
	return df

def previous_ret(df,col,n):
	df["previous_ret"] = (df[col].shift(int(n)) - df[col]) / df[col]
	return df


def k_enveloppe(df, n=10):
	df[f"EMA_HIGH_{n}"] = df["high"].ewm(span=n).mean()
	df[f"EMA_LOW_{n}"] = df["low"].ewm(span=n).mean()

	df["pivots_high"] = (df["close"] - df[f"EMA_HIGH_{n}"])/ df[f"EMA_HIGH_{n}"]
	df["pivots_low"] = (df["close"] - df[f"EMA_LOW_{n}"])/ df[f"EMA_LOW_{n}"]
	df["pivots"] = (df["pivots_high"]+df["pivots_low"])/2
	return df

def candle_information(df):
	# Candle color
	df["candle_way"] = -1
	df.loc[(df["open"] - df["close"]) < 0, "candle_way"] = 1

	# Filling percentage
	df["filling"] = np.abs(df["close"] - df["open"]) / np.abs(df["high"] - df["low"])

	# Amplitude
	df["amplitude"] = np.abs(df["close"] - df["open"]) / (df["open"] / 2 + df["close"] / 2) * 100

	return df

def data_split(df_model, split, list_X, list_y):

	# Train set creation
	X_train = df_model[list_X].iloc[0:split-1, :]
	y_train = df_model[list_y].iloc[1:split]

	# Test set creation
	X_test = df_model[list_X].iloc[split:-1, :]
	y_test = df_model[list_y].iloc[split+1:]

	return X_train, X_test, y_train, y_test

def quantile_signal(df, n, quantile_level=0.67,pct_split=0.8):

	n = int(n)

	# Create the split between train and test set to do not create a Look-Ahead bais
	split = int(len(df) * pct_split)

	# Copy the dataframe to do not create any intereference
	df_copy = df.copy()

	# Create the fut_ret column to be able to create the signal
	df_copy["fut_ret"] = (df_copy["close"].shift(-n) - df_copy["open"]) / df_copy["open"]

	# Create a column by name, 'Signal' and initialize with 0
	df_copy['Signal'] = 0

	# Assign a value of 1 to 'Signal' column for the quantile with the highest returns
	df_copy.loc[df_copy['fut_ret'] > df_copy['fut_ret'][:split].quantile(q=quantile_level), 'Signal'] = 1

	# Assign a value of -1 to 'Signal' column for the quantile with the lowest returns
	df_copy.loc[df_copy['fut_ret'] < df_copy['fut_ret'][:split].quantile(q=1-quantile_level), 'Signal'] = -1

	return df_copy

def binary_signal(df, n):

	n = int(n)

	# Copy the dataframe to do not create any intereference
	df_copy = df.copy()

	# Create the fut_ret column to be able to create the signal
	df_copy["fut_ret"] = (df_copy["close"].shift(-n) - df_copy["open"]) / df_copy["open"]

	# Create a column by name, 'Signal' and initialize with 0
	df_copy['Signal'] = -1

	# Assign a value of 1 to 'Signal' column for the quantile with the highest returns
	df_copy.loc[df_copy['fut_ret'] > 0, 'Signal'] = 1

	return df_copy

"""these are all the signals from alphas on machine learning"""
def rank(df):
	"""Return the cross-sectional percentile rank

		Args:
			:param df: tickers in columns, sorted dates in rows.

		Returns:
			pd.DataFrame: the ranked values
		"""
	return df.rank(axis=1, pct=True)

def scale(df):
	"""
	Scaling time serie.
	:param df: a pandas DataFrame.
	:param k: scaling factor.
	:return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
	"""
	return df.div(df.abs().sum(axis=1), axis=0)

def log(df):
	return np.log1p(df)
def sign(df):
	return np.sign(df)
def power(df, exp):
	return df.pow(exp)

def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
	"""Return the lagged values t periods ago.

	Args:
		:param df: tickers in columns, sorted dates in rows.
		:param t: lag

	Returns:
		pd.DataFrame: the lagged values
	"""
	return df.shift(t)

def ts_delta(df, period=1):
	"""
	Wrapper function to estimate difference.
	:param df: a pandas DataFrame.
	:param period: the difference grade.
	:return: a pandas DataFrame with today’s value minus the value 'period' days ago.
	"""
	return df.diff(period)

def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
	"""Computes the rolling ts_sum for the given window size.
	Args:
		df (pd.DataFrame): tickers in columns, dates in rows.
		window      (int): size of rolling window.
	Returns:
		pd.DataFrame: the ts_sum over the last 'window' days.
	"""
	return df.rolling(window).sum()

def ts_mean(df, window=10):
	"""Computes the rolling mean for the given window size.
	Args:
		df (pd.DataFrame): tickers in columns, dates in rows.
		window      (int): size of rolling window.

	Returns:
		pd.DataFrame: the mean over the last 'window' days.
	"""
	return df.rolling(window).mean()

def ts_weighted_mean(df, period=10):
	"""
	Linear weighted moving average implementation.
	:param df: a pandas DataFrame.
	:param period: the LWMA period
	:return: a pandas DataFrame with the LWMA.
	"""
	return (df.apply(lambda x: WMA(x, timeperiod=period)))

def ts_std(df, window=10):
	"""
	Wrapper function to estimate rolling standard deviation.
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: a pandas DataFrame with the time-series min over the past 'window' days.
	"""
	return (df.rolling(window).std())

def ts_rank(df, window=10):
	"""
	Wrapper function to estimate rolling rank.
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: a pandas DataFrame with the time-series rank over the past window days.
	"""
	return (df.rolling(window).apply(lambda x: x.rank().iloc[-1]))

def ts_product(df, window=10):
	"""
	Wrapper function to estimate rolling ts_product.
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: a pandas DataFrame with the time-series ts_product over the past 'window' days.
	"""
	return (df.rolling(window).apply(np.prod))

def ts_min(df, window=10):
	"""
	Wrapper function to estimate rolling min.
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: a pandas DataFrame with the time-series min over the past 'window' days.
	"""
	return df.rolling(window).min()

def ts_max(df, window=10):
	"""
	Wrapper function to estimate rolling min.
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: a pandas DataFrame with the time-series max over the past 'window' days.
	"""
	return df.rolling(window).max()

def ts_argmax(df, window=10):
	"""
	Wrapper function to estimate which day ts_max(df, window) occurred on
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: well.. that :)
	"""
	return df.rolling(window).apply(np.argmax).add(1)

def ts_argmin(df, window=10):
	"""
	Wrapper function to estimate which day ts_min(df, window) occurred on
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: well.. that :)
	"""
	return (df.rolling(window).apply(np.argmin).add(1))

def ts_corr(x, y, window=10):
	"""
	Wrapper function to estimate rolling correlations.
	:param x, y: pandas DataFrames.
	:param window: the rolling window.
	:return: a pandas DataFrame with the time-series min over the past 'window' days.
	"""
	return x.rolling(window).corr(y)

def ts_cov(x, y, window=10):
	"""
	Wrapper function to estimate rolling covariance.
	:param df: a pandas DataFrame.
	:param window: the rolling window.
	:return: a pandas DataFrame with the time-series min over the past 'window' days.
	"""
	return x.rolling(window).cov(y)

def alpha001(c, r):
	"""(rank(ts_argmax(power(((returns < 0)
		? ts_std(returns, 20)
		: close), 2.), 5)) -0.5)"""
	c[r < 0] = ts_std(r, 20)
	return (rank(ts_argmax(power(c, 2), 5)).mul(-.5).stack().swaplevel())

def alpha002(o, c, v):
	"""(-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
	s1 = rank(ts_delta(log(v), 2))
	s2 = rank((c / o) - 1)
	alpha = -ts_corr(s1, s2, 6)
	return alpha.stack('ticker').swaplevel().replace([-np.inf, np.inf], np.nan)

def alpha003(o, v):
	"""(-1 * ts_corr(rank(open), rank(volume), 10))"""

	return (-ts_corr(rank(o), rank(v), 10).stack('ticker').swaplevel().replace([-np.inf, np.inf], np.nan))

def alpha004(l):
	"""(-1 * Ts_Rank(rank(low), 9))"""
	return (-ts_rank(rank(l), 9).stack('ticker').swaplevel())

def alpha005(o, vwap, c):
	"""(rank((open - ts_mean(vwap, 10))) * (-1 * abs(rank((close - vwap)))))"""
	return (rank(o.sub(ts_mean(vwap, 10))).mul(rank(c.sub(vwap)).mul(-1).abs())
			.stack('ticker')
			.swaplevel())
	
def alpha006(df, o, v):
	o = df['Open']
	v = df['Volume']
	"""(-ts_corr(open, volume, 10))"""
	df['alpha006'] = (-ts_corr(o, v, 10))
	
def alpha007(self, df, c, v, adv20):
	"""(adv20 < volume)
		? ((-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7)))
		: -1
	"""
	delta7 = self.ts_delta(c, 7)
	df['alpha007'] = -self.ts_rank(abs(delta7), 60).mul(self.sign(delta7)).where(adv20<v, -1)

def alpha008(o, r):
	"""-rank(((ts_sum(open, 5) * ts_sum(returns, 5)) -
		ts_lag((ts_sum(open, 5) * ts_sum(returns, 5)),10)))
	"""
	return (-(rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
						ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10))))
			.stack('ticker')
			.swaplevel())
	
def alpha009(df, c):
	"""(0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1)
	: ((ts_max(ts_delta(close, 1), 5) < 0)
	? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))
	"""
	close_diff = ts_delta(c, 1)
	df['alpha009'] = close_diff.where(ts_min(close_diff, 5) > 0,
								close_diff.where(ts_max(close_diff, 5) < 0,
												-close_diff))

def alpha010(self, df, c):
	"""rank(((0 < ts_min(ts_delta(close, 1), 4))
		? ts_delta(close, 1)
		: ((ts_max(ts_delta(close, 1), 4) < 0)
			? ts_delta(close, 1)
			: (-1 * ts_delta(close, 1)))))
	"""
	close_diff = self.ts_delta(c, 1)
	alpha = close_diff.where(self.ts_min(close_diff, 4) > 0,
								close_diff.where(self.ts_min(close_diff, 4) > 0,
												-close_diff))
	return (rank(alpha).stack('ticker').swaplevel())

def alpha011(c, vwap, v):
	"""(rank(ts_max((vwap - close), 3)) +
		rank(ts_min(vwap - close), 3)) *
		rank(ts_delta(volume, 3))
		"""
	return (rank(ts_max(vwap.sub(c), 3)).add(rank(ts_min(vwap.sub(c), 3))).mul(rank(ts_delta(v, 3))).stack('ticker').swaplevel())

def alpha012(self, df, v, c):
	"""(sign(ts_delta(volume, 1)) *
			(-1 * ts_delta(close, 1)))
		"""
	df['alpha012'] = (sign(ts_delta(v, 1)).mul(-ts_delta(c, 1)))

def alpha021(df, c, v):
	"""ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
		? -1
		: (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
			? 1
			: (volume / adv20 < 1
				? -1
				: 1))
	"""
	sma2 = ts_mean(c, 2)
	sma8 = ts_mean(c, 8)
	std8 = ts_std(c, 8)

	cond_1 = sma8.add(std8) < sma2
	cond_2 = sma8.add(std8) > sma2
	cond_3 = v.div(ts_mean(v, 20)) < 1

	val = np.ones_like(c)
	df['alpha021'] = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3], choicelist=[-1, 1, -1], default=1))

def alpha023(df, h, c):
	"""((ts_mean(high, 20) < high)
			? (-1 * ts_delta(high, 2))
			: 0
		"""
	df['alpha023'] = (ts_delta(h, 2).mul(-1).where(ts_mean(h, 20) < h, 0))

def alpha024(self, df, c):
	"""((((ts_delta((ts_mean(close, 100)), 100) / ts_lag(close, 100)) <= 0.05)
		? (-1 * (close - ts_min(close, 100)))
		: (-1 * ts_delta(close, 3)))
	"""
	cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05

	return (c.sub(ts_min(c, 100)).mul(-1).where(cond, -ts_delta(c, 3))
			.stack('ticker')
			.swaplevel())
	
def alpha026(df, h, v):
	"""(-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
	df['alpha026'] = ts_max(ts_corr(ts_rank(v, 5), ts_rank(h, 5), 5).replace([-np.inf, np.inf], np.nan), 3).mul(-1)

def alpha028(df, h, l, c, v, adv20):
	"""scale(((ts_corr(adv20, low, 5) + (high + low) / 2) - close))"""
	df['alpha028'] = scale(ts_corr(adv20, l, 5).replace([-np.inf, np.inf], 0).add(h.add(l).div(2).sub(c)))

def alpha032(df, c, vwap):
	"""scale(ts_mean(close, 7) - close) +
		(20 * scale(ts_corr(vwap, ts_lag(close, 5),230)))"""
	df['alpha032'] = scale(ts_mean(c, 7).sub(c)).add(20 * scale(ts_corr(vwap, ts_lag(c, 5), 230)))

def alpha035(df, h, l, c, v, r):
	"""((ts_Rank(volume, 32) *
		(1 - ts_Rank(((close + high) - low), 16))) *
		(1 -ts_Rank(returns, 32)))
	"""
	df['alpha035'] = ts_rank(v, 32).mul(1 - ts_rank(c.add(h).sub(l), 16)).mul(1 - ts_rank(r, 32))

def alpha041(df, h, l, vwap):
	"""power(high * low, 0.5 - vwap"""
	df['alpha041'] = power(h.mul(l), 0.5).sub(vwap)

def alpha043(df, c, adv20, v):
	"""(ts_rank((volume / adv20), 20) * ts_rank((-1 * ts_delta(close, 7)), 8))"""

	df['alpha043'] = ts_rank(v.div(adv20), 20).mul(ts_rank(ts_delta(c, 7).mul(-1), 8))

def alpha044(h, v):
	"""-ts_corr(high, rank(volume), 5)"""
	df['alpha044'] = ts_corr(h, rank(v), 5).replace([-np.inf, np.inf], np.nan).mul(-1)

def alpha046(df, c):
	"""0.25 < ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10
			? -1
			: ((ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10 < 0)
				? 1
				: -ts_delta(close, 1))
	"""
	cond = ts_lag(ts_delta(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
	alpha = pd.DataFrame(-np.ones_like(cond))
	alpha[cond.isnull()] = np.nan
	df['alpha046'] = cond.where(cond > 0.25, -alpha.where(cond < 0, -ts_delta(c, 1)))



def alpha050(df, v, vwap):
	"""-ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)"""
	return (ts_max(rank(ts_corr(rank(v),
                             rank(vwap), 5)), 5).mul(-1))

def alpha051(df, c):
	"""ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.05 * c
		? 1
		: -ts_delta(close, 1)"""
	cond = (ts_delta(ts_lag(c, 10), 10).div(10)
			.sub(ts_delta(c, 10).div(10)) >= -0.05 * c)
	df['alpha051']  = -ts_delta(c, 1).where(cond, 1)

def alpha053(df, h, l, c):
	"""-1 * ts_delta(1 - (high - close) / (close - low), 9)"""
	inner = (c.sub(l)).add(1e-6)
	df['alpha053'] = ts_delta(h.sub(c).mul(-1).add(1).div(c.sub(l).add(1e-6)), 9).mul(-1)

def alpha054(df, o, h, l, c):
	"""-(low - close) * power(open, 5) / ((low - high) * power(close, 5))"""
	df['alpha054'] = l.sub(c).mul(o.pow(5)).mul(-1).div(l.sub(h).replace(0, -0.0001).mul(c ** 5))

def alpha057(c, vwap):
	"""-(close - vwap) / ts_weighted_mean(rank(ts_argmax(close, 30)), 2)"""
	return (c.sub(vwap.add(1e-5))
			.div(ts_weighted_mean(rank(ts_argmax(c, 30)))).mul(-1)
			.stack('ticker')
			.swaplevel())
	
def alpha086(c, v, vwap):
	"""((ts_rank(ts_corr(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195) <
		rank(((open + close) - (vwap + open)))) * -1)
	"""
	df['alpha086']  = ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20).lt(rank(c.sub(vwap))).mul(-1)

def alpha092(df, o, h, l, c, v):
	"""min(ts_rank(ts_weighted_mean(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683),
			ts_rank(ts_weighted_mean(ts_corr(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
	"""
	p1 = ts_rank(ts_weighted_mean(h.add(l).div(2).add(c).lt(l.add(o)), 15), 18)
	p2 = ts_rank(ts_weighted_mean(ts_corr(rank(l), rank(ts_mean(v, 30)), 7), 6), 6)

	df['alpha092'] =  p1.where(p1<p2, p2)

def alpha096(c, v, vwap):
	"""(max(ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(volume), 5.83878),4.16783), 8.38151),
		ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)"""

	s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)
	s2 = ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(c, 7),
													ts_rank(ts_mean(v, 60), 10), 10), 12), 14), 13)
	return (s1.where(s1 > s2, s2)
			.mul(-1)
			.stack('ticker')
			.swaplevel())

def alpha101(df, o, h, l, c):
	"""((close - open) / ((high - low) + .001))"""
	df['alpha101'] = c.sub(o).div(h.sub(l).add(1e-3))
