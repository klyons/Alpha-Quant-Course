import ta
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.stattools import adfuller


def sma(df, col, n):
    df[f"SMA_{n}"] = ta.trend.SMAIndicator(df[col],int(n)).sma_indicator()
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

def garch_prediction(df, col):
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
    n : int
        The window period for KAMA calculation.
    
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
