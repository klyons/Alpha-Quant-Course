import ta
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from fracdiff.sklearn import Fracdiff, FracdiffStat
import pdb
# https://gist.github.com/jmrichardson/43ec7d21faa775f2f6c6ee46dc611655


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

def alpha_02(df, periods=5):
    """
    Calculate the alpha -delta(close, periods).

    Parameters:
    df (pd.DataFrame): DataFrame containing 'close' column.
    periods (int): Number of periods to look back. Default is 5.

    Returns:
    pd.DataFrame: DataFrame with the alpha added as a new column.
    """
    df_copy = df.copy()
    df_copy['alpha_02'] = -(df_copy['close'] - df_copy['close'].shift(periods))
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




def rank(df):
    return df.rank(axis=1, pct=True)


def scale(df):
    return df.div(df.abs().sum(axis=1), axis=0)


def log(df):
    return np.log1p(df)


def sign(df):
    return np.sign(df)


def power(df, exp):
    return df.pow(exp)


def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    return df.shift(t)


def ts_delta(df, period=1):
    return df.diff(period)


def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).sum()


def ts_mean(df, window=10):
    return df.rolling(window).mean()


def ts_weighted_mean(df, period=10):
    return (df.apply(lambda x: WMA(x, timeperiod=period)))


def ts_std(df, window=10):
    return (df
            .rolling(window)
            .std())


def ts_rank(df, window=10):
    return (df
            .rolling(window)
            .apply(lambda x: x.rank().iloc[-1]))


def ts_product(df, window=10):
    return (df
            .rolling(window)
            .apply(np.prod))


def ts_min(df, window=10):
    return df.rolling(window).min()


def ts_max(df, window=10):
    return df.rolling(window).max()


def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax).add(1)


def ts_argmin(df, window=10):
    return (df.rolling(window)
            .apply(np.argmin)
            .add(1))


def ts_corr(x, y, window):
    res = Parallel(n_jobs=cpu_count())(delayed(rolling_apply)(dcor, window, x[col], y[col]) for col in x)
    res = pd.DataFrame.from_dict(dict(zip(x.columns, res)))
    res.index = x.index
    res.columns.name = 'ticker'
    return res

# def ts_corr(x, y, window=10):
    # res = x.rolling(window).corr(y)
    # return res

def ts_cov(x, y, window=10):
    return x.rolling(window).cov(y)


def get_mutual_info_score(returns, alpha, n=100000):
    df = pd.DataFrame({'y': returns, 'alpha': alpha}).dropna().sample(n=n)
    return mutual_info_regression(y=df.y, X=df[['alpha']])[0]


def alpha001(c, r):
    c[r < 0] = ts_std(r, 20)
    return (rank(ts_argmax(power(c, 2), 5)).mul(-.5)
            .stack().swaplevel())


def alpha002(o, c, v):
    s1 = rank(ts_delta(log(v), 2))
    s2 = rank((c / o) - 1)
    alpha = -ts_corr(s1, s2, 6)
    res = alpha.stack('ticker').swaplevel().replace([-np.inf, np.inf], np.nan)
    return res


def alpha003(o, v):
    return (-ts_corr(rank(o), rank(v), 10)
            .stack('ticker')
            .swaplevel()
            .replace([-np.inf, np.inf], np.nan))


def alpha004(l):
    return (-ts_rank(rank(l), 9)
            .stack('ticker')
            .swaplevel())


def alpha005(o, vwap, c):
    return (rank(o.sub(ts_mean(vwap, 10)))
            .mul(rank(c.sub(vwap)).mul(-1).abs())
            .stack('ticker')
            .swaplevel())


def alpha006(o, v):
    return (-ts_corr(o, v, 10)
            .stack('ticker')
            .swaplevel())


def alpha007(c, v, adv20):
    delta7 = ts_delta(c, 7)
    return (-ts_rank(abs(delta7), 60)
            .mul(sign(delta7))
            .where(adv20 < v, -1)
            .stack('ticker')
            .swaplevel())


def alpha008(o, r):
    return (-(rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
                    ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10))))
            .stack('ticker')
            .swaplevel())


def alpha009(c):
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                             close_diff.where(ts_max(close_diff, 5) < 0,
                                              -close_diff))
    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha010(c):
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 4) > 0,
                             close_diff.where(ts_min(close_diff, 4) > 0,
                                              -close_diff))

    return (rank(alpha)
            .stack('ticker')
            .swaplevel())


def alpha011(c, vwap, v):
    return (rank(ts_max(vwap.sub(c), 3))
            .add(rank(ts_min(vwap.sub(c), 3)))
            .mul(rank(ts_delta(v, 3)))
            .stack('ticker')
            .swaplevel())


def alpha012(v, c):
    return (sign(ts_delta(v, 1)).mul(-ts_delta(c, 1))
            .stack('ticker')
            .swaplevel())


def alpha013(c, v):
    return (-rank(ts_cov(rank(c), rank(v), 5))
            .stack('ticker')
            .swaplevel())


def alpha014(o, v, r):
    alpha = -rank(ts_delta(r, 3)).mul(ts_corr(o, v, 10)
                                      .replace([-np.inf,
                                                np.inf],
                                               np.nan))
    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha015(h, v):
    alpha = (-ts_sum(rank(ts_corr(rank(h), rank(v), 3)
                          .replace([-np.inf, np.inf], np.nan)), 3))
    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha016(h, v):
    return (-rank(ts_cov(rank(h), rank(v), 5))
            .stack('ticker')
            .swaplevel())


def alpha017(c, v):
    adv20 = ts_mean(v, 20)
    return (-rank(ts_rank(c, 10))
            .mul(rank(ts_delta(ts_delta(c, 1), 1)))
            .mul(rank(ts_rank(v.div(adv20), 5)))
            .stack('ticker')
            .swaplevel())


def alpha018(o, c):
    return (-rank(ts_std(c.sub(o).abs(), 5)
                  .add(c.sub(o))
                  .add(ts_corr(c, o, 10)
                       .replace([-np.inf,
                                 np.inf],
                                np.nan)))
            .stack('ticker')
            .swaplevel())


def alpha019(c, r):
    return (-sign(ts_delta(c, 7) + ts_delta(c, 7))
            .mul(1 + rank(1 + ts_sum(r, 250)))
            .stack('ticker')
            .swaplevel())


def alpha020(o, h, l, c):
    return (rank(o - ts_lag(h, 1))
            .mul(rank(o - ts_lag(c, 1)))
            .mul(rank(o - ts_lag(l, 1)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha021(c, v):
    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)

    cond_1 = sma8.add(std8) < sma2
    cond_2 = sma8.add(std8) > sma2
    cond_3 = v.div(ts_mean(v, 20)) < 1

    val = np.ones_like(c)
    alpha = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3],
                                   choicelist=[-1, 1, -1], default=1),
                         index=c.index,
                         columns=c.columns)

    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha022(h, c, v):
    return (ts_delta(ts_corr(h, v, 5)
                     .replace([-np.inf,
                               np.inf],
                              np.nan), 5)
            .mul(rank(ts_std(c, 20)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha023(h, c):
    return (ts_delta(h, 2)
            .mul(-1)
            .where(ts_mean(h, 20) < h, 0)
            .stack('ticker')
            .swaplevel())


def alpha024(c):
    cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05

    return (c.sub(ts_min(c, 100)).mul(-1).where(cond, -ts_delta(c, 3))
            .stack('ticker')
            .swaplevel())


def alpha025(h, c, r, vwap, adv20):
    return (rank(-r.mul(adv20)
                 .mul(vwap)
                 .mul(h.sub(c)))
            .stack('ticker')
            .swaplevel())


def alpha026(h, v):
    return (ts_max(ts_corr(ts_rank(v, 5),
                           ts_rank(h, 5), 5)
                   .replace([-np.inf, np.inf], np.nan), 3)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha027(v, vwap):
    cond = rank(ts_mean(ts_corr(rank(v),
                                rank(vwap), 6), 2))
    alpha = cond.notnull().astype(float)
    return (alpha.where(cond <= 0.5, -alpha)
            .stack('ticker')
            .swaplevel())


def alpha028(h, l, c, v, adv20):
    return (scale(ts_corr(adv20, l, 5)
                  .replace([-np.inf, np.inf], 0)
                  .add(h.add(l).div(2).sub(c)))
            .stack('ticker')
            .swaplevel())


def alpha029(c, r):
    return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta((c - 1), 5)))), 2))))), 5)
            .add(ts_rank(ts_lag((-1 * r), 6), 5))
            .stack('ticker')
            .swaplevel())


def alpha030(c, v):
    close_diff = ts_delta(c, 1)
    return (rank(sign(close_diff)
                 .add(sign(ts_lag(close_diff, 1)))
                 .add(sign(ts_lag(close_diff, 2))))
            .mul(-1).add(1)
            .mul(ts_sum(v, 5))
            .div(ts_sum(v, 20))
            .stack('ticker')
            .swaplevel())


def alpha031(l, c, adv20):
    return (rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(c, 10))).mul(-1), 10))))
            .add(rank(ts_delta(c, 3).mul(-1)))
            .add(sign(scale(ts_corr(adv20, l, 12)
                            .replace([-np.inf, np.inf],
                                     np.nan))))
            .stack('ticker')
            .swaplevel())


def alpha032(c, vwap):
    return (scale(ts_mean(c, 7).sub(c))
            .add(20 * scale(ts_corr(vwap,
                                    ts_lag(c, 5), 230)))
            .stack('ticker')
            .swaplevel())


def alpha033(o, c):
    return (rank(o.div(c).mul(-1).add(1).mul(-1))
            .stack('ticker')
            .swaplevel())


def alpha034(c, r):
    return (rank(rank(ts_std(r, 2).div(ts_std(r, 5))
                      .replace([-np.inf, np.inf],
                               np.nan))
                 .mul(-1)
                 .sub(rank(ts_delta(c, 1)))
                 .add(2))
            .stack('ticker')
            .swaplevel())


def alpha035(h, l, c, v, r):
    return (ts_rank(v, 32)
            .mul(1 - ts_rank(c.add(h).sub(l), 16))
            .mul(1 - ts_rank(r, 32))
            .stack('ticker')
            .swaplevel())


def alpha036(o, c, v, r, adv20, vwap):
    return (rank(ts_corr(c.sub(o), ts_lag(v, 1), 15)).mul(2.21)
            .add(rank(o.sub(c)).mul(.7))
            .add(rank(ts_rank(ts_lag(-r, 6), 5)).mul(0.73))
            .add(rank(abs(ts_corr(vwap, adv20, 6))))
            .add(rank(ts_mean(c, 200).sub(o).mul(c.sub(o))).mul(0.6))
            .stack('ticker')
            .swaplevel())


def alpha037(o, c):
    return (rank(ts_corr(ts_lag(o.sub(c), 1), c, 200))
            .add(rank(o.sub(c)))
            .stack('ticker')
            .swaplevel())


def alpha038(o, c):
    return (rank(ts_rank(o, 10))
            .mul(rank(c.div(o).replace([-np.inf, np.inf], np.nan)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha039(c, v, r, adv20):
    return (rank(ts_delta(c, 7).mul(rank(ts_weighted_mean(v.div(adv20), 9)).mul(-1).add(1))).mul(-1)
            .mul(rank(ts_mean(r, 250).add(1)))
            .stack('ticker')
            .swaplevel())


def alpha040(h, v):
    return (rank(ts_std(h, 10))
            .mul(ts_corr(h, v, 10))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha041(h, l, vwap):
    return (power(h.mul(l), 0.5)
            .sub(vwap)
            .stack('ticker')
            .swaplevel())


def alpha042(c, vwap):
    return (rank(vwap.sub(c))
            .div(rank(vwap.add(c)))
            .stack('ticker')
            .swaplevel())


def alpha043(c, v, adv20):
    return (ts_rank(v.div(adv20), 20)
            .mul(ts_rank(ts_delta(c, 7).mul(-1), 8))
            .stack('ticker')
            .swaplevel())


def alpha044(h, v):
    return (ts_corr(h, rank(v), 5)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha045(c, v):
    return (rank(ts_mean(ts_lag(c, 5), 20))
            .mul(ts_corr(c, v, 2)
                 .replace([-np.inf, np.inf], np.nan))
            .mul(rank(ts_corr(ts_sum(c, 5),
                              ts_sum(c, 20), 2)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha046(c):
    cond = ts_lag(ts_delta(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
    alpha = pd.DataFrame(-np.ones_like(cond),
                         index=c.index,
                         columns=c.columns)
    alpha[cond.isnull()] = np.nan
    return (cond.where(cond > 0.25,
                       -alpha.where(cond < 0,
                                    -ts_delta(c, 1)))
            .stack('ticker')
            .swaplevel())


def alpha047(h, c, v, vwap, adv20):
    return (rank(c.pow(-1)).mul(v).div(adv20)
            .mul(h.mul(rank(h.sub(c))
                       .div(ts_mean(h, 5)))
                 .sub(rank(ts_delta(vwap, 5))))
            .stack('ticker')
            .swaplevel())


def alpha049(c):
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.1 * c)
    return (-ts_delta(c, 1)
            .where(cond, 1)
            .stack('ticker')
            .swaplevel())


def alpha050(v, vwap):
    return (ts_max(rank(ts_corr(rank(v),
                                rank(vwap), 5)), 5)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha051(c):
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.05 * c)
    return (-ts_delta(c, 1)
            .where(cond, 1)
            .stack('ticker')
            .swaplevel())



def alpha052(l, v, r):
    return (ts_delta(ts_min(l, 5), 5)
            .mul(rank(ts_sum(r, 240)
                      .sub(ts_sum(r, 20))
                      .div(220)))
            .mul(ts_rank(v, 5))
            .stack('ticker')
            .swaplevel())


def alpha053(h, l, c):
    inner = (c.sub(l)).add(1e-6)
    return (ts_delta(h.sub(c)
                     .mul(-1).add(1)
                     .div(c.sub(l)
                          .add(1e-6)), 9)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha054(o, h, l, c):
    return (l.sub(c).mul(o.pow(5)).mul(-1)
            .div(l.sub(h).replace(0, -0.0001).mul(c ** 5))
            .stack('ticker')
            .swaplevel())


def alpha055(h, l, c, v):
    return (ts_corr(rank(c.sub(ts_min(l, 12))
                         .div(ts_max(h, 12).sub(ts_min(l, 12))
                              .replace(0, 1e-6))),
                    rank(v), 6)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha057(c, vwap):
    return (c.sub(vwap.add(1e-5))
            .div(ts_weighted_mean(rank(ts_argmax(c, 30)))).mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha060(l, h, c, v):
    return (scale(rank(c.mul(2).sub(l).sub(h)
                       .div(h.sub(l).replace(0, 1e-5))
                       .mul(v))).mul(2)
            .sub(scale(rank(ts_argmax(c, 10)))).mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha061(v, vwap):
    return (rank(vwap.sub(ts_min(vwap, 16)))
            .lt(rank(ts_corr(vwap, ts_mean(v, 180), 18)))
            .astype(int)
            .stack('ticker')
            .swaplevel())


def alpha062(o, h, l, vwap, adv20):
    return (rank(ts_corr(vwap, ts_sum(adv20, 22), 9))
            .lt(rank(
        rank(o).mul(2))
                .lt(rank(h.add(l).div(2))
                    .add(rank(h))))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha064(o, h, l, v, vwap):
    w = 0.178404
    return (rank(ts_corr(ts_sum(o.mul(w).add(l.mul(1 - w)), 12),
                         ts_sum(ts_mean(v, 120), 12), 16))
            .lt(rank(ts_delta(h.add(l).div(2).mul(w)
                              .add(vwap.mul(1 - w)), 3)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha065(o, v, vwap):
    w = 0.00817205
    return (rank(ts_corr(o.mul(w).add(vwap.mul(1 - w)),
                         ts_mean(ts_mean(v, 60), 9), 6))
            .lt(rank(o.sub(ts_min(o, 13))))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha066(o, l, h, vwap):
    w = 0.96633
    return (rank(ts_weighted_mean(ts_delta(vwap, 4), 7))
            .add(ts_rank(ts_weighted_mean(l.mul(w).add(l.mul(1 - w))
                                          .sub(vwap)
                                          .div(o.sub(h.add(l).div(2)).add(1e-3)), 11), 7))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha068(h, l, c, v):
    w = 0.518371
    return (ts_rank(ts_corr(rank(h), rank(ts_mean(v, 15)), 9), 14)
            .lt(rank(ts_delta(c.mul(w).add(l.mul(1 - w)), 1)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha071(o, l, c, v, vwap):
    s1 = (ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 3),
                                           ts_rank(ts_mean(v, 180), 12), 18), 4), 16))
    s2 = (ts_rank(ts_weighted_mean(rank(l.add(o).
                                        sub(vwap.mul(2)))
                                   .pow(2), 16), 4))
    return (s1.where(s1 > s2, s2)
            .stack('ticker')
            .swaplevel())


def alpha072(h, l, v, vwap):
    return (rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 9), 10))
            .div(rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3), ts_rank(v, 18), 6), 2)))
            .stack('ticker')
            .swaplevel())


def alpha073(o, l, vwap):
    w = 0.147155
    s1 = rank(ts_weighted_mean(ts_delta(vwap, 5), 3))
    s2 = (ts_rank(ts_weighted_mean(ts_delta(o.mul(w).add(l.mul(1 - w)), 2)
                                   .div(o.mul(w).add(l.mul(1 - w)).mul(-1)), 3), 16))

    return (s1.where(s1 > s2, s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha074(h, c, v, vwap):
    w = 0.0261661
    return (rank(ts_corr(c, ts_mean(ts_mean(v, 30), 37), 15))
            .lt(rank(ts_corr(rank(h.mul(w).add(vwap.mul(1 - w))), rank(v), 11)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha075(l, v, vwap):
    return (rank(ts_corr(vwap, v, 4))
            .lt(rank(ts_corr(rank(l), rank(ts_mean(v, 50)), 12)))
            .astype(int)
            .stack('ticker')
            .swaplevel())


def alpha077(l, h, v, vwap):
    s1 = rank(ts_weighted_mean(h.add(l).div(2).sub(vwap), 20))
    s2 = rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 3), 5))
    return (s1.where(s1 < s2, s2)
            .stack('ticker')
            .swaplevel())


def alpha078(l, v, vwap):
    w = 0.352233
    return (rank(ts_corr(ts_sum((l.mul(w).add(vwap.mul(1 - w))), 19),
                         ts_sum(ts_mean(v, 40), 19), 6))
            .pow(rank(ts_corr(rank(vwap), rank(v), 5)))
            .stack('ticker')
            .swaplevel())


def alpha081(v, vwap):
    return (rank(log(ts_product(rank(rank(ts_corr(vwap,
                                                  ts_sum(ts_mean(v, 10), 50), 8))
                                     .pow(4)), 15)))
            .lt(rank(ts_corr(rank(vwap), rank(v), 5)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha083(h, l, c, v, vwap):
    s = h.sub(l).div(ts_mean(c, 5))

    return (rank(rank(ts_lag(s, 2))
                 .mul(rank(rank(v)))
                 .div(s).div(vwap.sub(c).add(1e-3)))
            .stack('ticker')
            .swaplevel()
            .replace((np.inf, -np.inf), np.nan))


def alpha084(c, vwap):
    return (rank(power(ts_rank(vwap.sub(ts_max(vwap, 15)), 20),
                       ts_delta(c, 6)))
            .stack('ticker')
            .swaplevel())


def alpha085(h, l, c, v):
    w = 0.876703
    return (rank(ts_corr(h.mul(w).add(c.mul(1 - w)), ts_mean(v, 30), 10))
            .pow(rank(ts_corr(ts_rank(h.add(l).div(2), 4),
                              ts_rank(v, 10), 7)))
            .stack('ticker')
            .swaplevel())


def alpha086(c, v, vwap):
    return (ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20)
            .lt(rank(c.sub(vwap)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha088(o, h, l, c, v):
    s1 = (rank(ts_weighted_mean(rank(o)
                                .add(rank(l))
                                .sub(rank(h))
                                .add(rank(c)), 8)))
    s2 = ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 8),
                                          ts_rank(ts_mean(v, 60), 20), 8), 6), 2)

    return (s1.where(s1 < s2, s2)
            .stack('ticker')
            .swaplevel())


def alpha092(o, h, l, c, v):
    p1 = ts_rank(ts_weighted_mean(h.add(l).div(2).add(c).lt(l.add(o)), 15), 18)
    p2 = ts_rank(ts_weighted_mean(ts_corr(rank(l), rank(ts_mean(v, 30)), 7), 6), 6)

    return (p1.where(p1 < p2, p2)
            .stack('ticker')
            .swaplevel())


def alpha094(v, vwap):
    return (rank(vwap.sub(ts_min(vwap, 11)))
            .pow(ts_rank(ts_corr(ts_rank(vwap, 20),
                                 ts_rank(ts_mean(v, 60), 4), 18), 2))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha095(o, h, l, v):
    return (rank(o.sub(ts_min(o, 12)))
            .lt(ts_rank(rank(ts_corr(ts_mean(h.add(l).div(2), 19),
                                     ts_sum(ts_mean(v, 40), 19), 13).pow(5)), 12))
            .astype(int)
            .stack('ticker')
            .swaplevel())


def alpha096(c, v, vwap):
    s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)
    s2 = ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(c, 7),
                                                    ts_rank(ts_mean(v, 60), 10), 10), 12), 14), 13)
    return (s1.where(s1 > s2, s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha098(o, v, vwap):
    adv5 = ts_mean(v, 5)
    adv15 = ts_mean(v, 15)
    return (rank(ts_weighted_mean(ts_corr(vwap, ts_mean(adv5, 26), 4), 7))
            .sub(rank(ts_weighted_mean(ts_rank(ts_argmin(ts_corr(rank(o),
                                                                 rank(adv15), 20), 8), 6))))
            .stack('ticker')
            .swaplevel())


def alpha099(h, l, v):
    return ((rank(ts_corr(ts_sum((h.add(l).div(2)), 19),
                          ts_sum(ts_mean(v, 60), 19), 8))
             .lt(rank(ts_corr(l, v, 6)))
             .mul(-1))
            .stack('ticker')
            .swaplevel())


def alpha101(o, h, l, c):
    return (c.sub(o).div(h.sub(l).add(1e-3))
            .stack('ticker')
            .swaplevel())

class Alphas101:

    def __init__(self):
        memory = Memory(f"store/{get_name()}", verbose=0)
        self.fit = memory.cache(self.fit)
        self.transform = memory.cache(self.transform)

    def fit(self, data):
        return self

    def transform(self, data):

        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        adv20 = data.groupby('ticker').rolling(20).volume.mean().reset_index(0, drop=True)
        data = data.assign(adv20=adv20)
        data = data.join(data.groupby('date')[ohlcv].rank(axis=1, pct=True), rsuffix='_rank')
        data.dropna(inplace=True)

        o = data.open.unstack('ticker')
        h = data.high.unstack('ticker')
        l = data.low.unstack('ticker')
        c = data.close.unstack('ticker')
        v = data.volume.unstack('ticker')
        vwap = o.add(h).add(l).add(c).div(4)
        adv20 = v.rolling(20).mean()
        r = data.close.unstack('ticker').pct_change()

        alphas = []
        alphas.append(alpha001(c, r).rename('alpha001'))
        alphas.append(alpha002(o, c, v).rename('alpha002'))
        alphas.append(alpha003(o, v).rename('alpha003'))
        alphas.append(alpha004(l).rename('alpha004'))
        alphas.append(alpha005(o, vwap, c).rename('alpha005'))
        alphas.append(alpha006(o, v).rename('alpha006'))
        alphas.append(alpha007(c, v, adv20).rename('alpha007'))
        alphas.append(alpha008(o, r).rename('alpha008'))
        alphas.append(alpha009(c).rename('alpha009'))
        alphas.append(alpha010(c).rename('alpha010'))
        alphas.append(alpha011(c, vwap, v).rename('alpha011'))
        alphas.append(alpha012(v, c).rename('alpha012'))
        alphas.append(alpha013(c, v).rename('alpha013'))
        alphas.append(alpha014(o, v, r).rename('alpha014'))
        alphas.append(alpha015(h, v).rename('alpha015'))
        alphas.append(alpha016(h, v).rename('alpha016'))
        alphas.append(alpha017(c, v).rename('alpha017'))
        alphas.append(alpha018(o, c).rename('alpha018'))
        alphas.append(alpha019(c, r).rename('alpha019'))
        alphas.append(alpha020(o, h, l, c).rename('alpha020'))
        alphas.append(alpha021(c, v).rename('alpha021'))
        alphas.append(alpha022(h, c, v).rename('alpha022'))
        alphas.append(alpha023(h, c).rename('alpha023'))
        alphas.append(alpha024(c).rename('alpha024'))
        alphas.append(alpha025(h, c, r, vwap, adv20).rename('alpha025'))
        alphas.append(alpha026(h, v).rename('alpha026'))
        alphas.append(alpha027(v, vwap).rename('alpha027'))
        alphas.append(alpha028(h, l, c, v, adv20).rename('alpha028'))
        alphas.append(alpha029(c, r).rename('alpha029'))
        alphas.append(alpha030(c, v).rename('alpha030'))
        # alphas.append(alpha031(l, c, adv20).rename('alpha031')) # Produces all nans
        alphas.append(alpha032(c, vwap).rename('alpha032'))
        alphas.append(alpha033(o, c).rename('alpha033'))
        alphas.append(alpha034(c, r).rename('alpha034'))
        alphas.append(alpha035(h, l, c, v, r).rename('alpha035'))
        alphas.append(alpha036(o, c, v, r, adv20, vwap).rename('alpha036'))
        alphas.append(alpha037(o, c).rename('alpha037'))
        alphas.append(alpha038(o, c).rename('alpha038'))
        alphas.append(alpha039(c, v, r, adv20).rename('alpha039'))
        alphas.append(alpha040(h, v).rename('alpha040'))
        alphas.append(alpha041(h, l, vwap).rename('alpha041'))
        alphas.append(alpha042(c, vwap).rename('alpha042'))
        alphas.append(alpha043(c, v, adv20).rename('alpha043'))
        alphas.append(alpha044(h, v).rename('alpha044'))
        alphas.append(alpha045(c, v).rename('alpha045'))
        alphas.append(alpha046(c).rename('alpha046'))
        alphas.append(alpha047(h, c, v, vwap, adv20).rename('alpha047'))
        # alphas.append(alpha048(h, c, vwap, adv20).rename('alpha048'))  # No implementation
        alphas.append(alpha049(c).rename('alpha049'))
        alphas.append(alpha050(v, vwap).rename('alpha050'))
        alphas.append(alpha051(c).rename('alpha051'))
        alphas.append(alpha052(l, v, r).rename('alpha052'))
        alphas.append(alpha053(h, l, c).rename('alpha053'))
        alphas.append(alpha054(o, h, l, c).rename('alpha054'))
        alphas.append(alpha055(h, l, c, v).rename('alpha055'))
        # alphas.append(alpha056(h, l, c).rename('alpha056'))  # No implementation
        # alphas.append(alpha057(h, l, c).rename('alpha057'))  # No implementation
        # alphas.append(alpha058(h, l, c).rename('alpha057'))  # No implementation
        # alphas.append(alpha059(h, l, c).rename('alpha059'))  # No implementation
        alphas.append(alpha060(l, h, c, v).rename('alpha060'))
        alphas.append(alpha061(v, vwap).rename('alpha061'))
        alphas.append(alpha062(o, h, l, vwap, adv20).rename('alpha062'))
        # alphas.append(alpha063(o, h, l, vwap, adv20).rename('alpha063'))  # No implementation
        alphas.append(alpha064(o, h, l, v, vwap).rename('alpha064'))
        alphas.append(alpha065(o, v, vwap).rename('alpha065'))
        alphas.append(alpha066(o, l, h, vwap).rename('alpha066'))
        # alphas.append(alpha067(l, h, vwap).rename('alpha067'))
        alphas.append(alpha068(h, l, c, v).rename('alpha068'))
        # alphas.append(alpha069(h, c, v).rename('alpha069'))
        # alphas.append(alpha070(h, c, v).rename('alpha070'))
        alphas.append(alpha071(o, l, c, v, vwap).rename('alpha071'))
        alphas.append(alpha072(h, l, v, vwap).rename('alpha072'))
        alphas.append(alpha073(o, l, vwap).rename('alpha073'))
        alphas.append(alpha074(h, c, v, vwap).rename('alpha074'))
        alphas.append(alpha075(l, v, vwap).rename('alpha075'))
        # alphas.append(alpha076(l, v, vwap).rename('alpha076'))
        alphas.append(alpha077(l, h, v, vwap).rename('alpha077'))
        alphas.append(alpha078(l, v, vwap).rename('alpha078'))
        # alphas.append(alpha079(l, v, vwap).rename('alpha079'))
        # alphas.append(alpha080(l, v, vwap).rename('alpha080'))
        alphas.append(alpha081(v, vwap).rename('alpha081'))
        # alphas.append(alpha082(v, vwap).rename('alpha082'))
        alphas.append(alpha083(h, l, c, v, vwap).rename('alpha083'))
        alphas.append(alpha084(c, vwap).rename('alpha084'))
        alphas.append(alpha085(h, l, c, v).rename('alpha085'))
        alphas.append(alpha086(c, v, vwap).rename('alpha086'))
        # alphas.append(alpha087(c, v, vwap).rename('alpha087'))
        alphas.append(alpha088(o, h, l, c, v).rename('alpha088'))
        # alphas.append(alpha089(o, h, l, c, v).rename('alpha089'))
        # alphas.append(alpha090(o, h, l, c, v).rename('alpha090'))
        # alphas.append(alpha091(o, h, l, c, v).rename('alpha091'))
        alphas.append(alpha092(o, h, l, c, v).rename('alpha092'))
        # alphas.append(alpha093(o, l, c, v).rename('alpha093'))
        alphas.append(alpha094(v, vwap).rename('alpha094'))
        alphas.append(alpha095(o, h, l, v).rename('alpha095'))
        alphas.append(alpha096(c, v, vwap).rename('alpha096'))
        # alphas.append(alpha097(c, v, vwap).rename('alpha097'))
        alphas.append(alpha098(o, v, vwap).rename('alpha098'))
        alphas.append(alpha099(h, l, v).rename('alpha099'))
        # alphas.append(alpha100(l, v).rename('alpha100'))
        alphas.append(alpha101(o, h, l, c).rename('alpha101'))

        features = pd.concat(alphas, axis=1)
        features = features.reorder_levels(order=[1, 0])
        features = features.sort_index()
        return features