from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5
import requests
import sys
import pdb

alphaVantageAPI = "G9ATZKK9US0QL6K8"

mt5.initialize()

print(mt5.account_info())



def get_save_path(symbol, timeframe):
    # Replace special characters in symbol with underscore
    symbol = symbol.replace("!", "_")
    
    # Convert timeframe to string
    if timeframe == mt5.TIMEFRAME_M1:
        timeframe_str = "1M"
    elif timeframe == mt5.TIMEFRAME_M5:
        timeframe_str = "5M"
    elif timeframe == mt5.TIMEFRAME_M10:
        timeframe_str = "10M"
    elif timeframe == mt5.TIMEFRAME_M15:
        timeframe_str = "15M"
    elif timeframe == mt5.TIMEFRAME_M30:
        timeframe_str = "M30"
    if timeframe == mt5.TIMEFRAME_H1:
        timeframe_str = "1H"
    elif timeframe == mt5.TIMEFRAME_H4:
        timeframe_str = "4H"
    
    # Add more conditions for other timeframes as needed
    save_path = f"C:/ws/Alpha-Quant-Course/Data/FixTimeBars/{symbol}{timeframe_str}.csv"
    
    return save_path

def get_rates(symbol, number_of_data=10_000, timeframe=mt5.TIMEFRAME_D1):
    # Compute now date
    from_date = datetime.now()

    # Extract n rates before now
    rates = mt5.copy_rates_from(symbol, timeframe, from_date, number_of_data)

    # Transform array into a DataFrame
    df_rates = pd.DataFrame(rates)
    # Convert number format of the date into date format
    df_rates["time"] = pd.to_datetime(df_rates["time"], unit="s")

    df_rates = df_rates.set_index("time")

    return df_rates

def get_ticks(symbol, number_of_data=10_000, timeframe=mt5.TIMEFRAME_D1):
    # Compute now date
    from_date = datetime.now()

    # Extract n rates before now
    ticks = mt5.copy_ticks_from(symbol, timeframe, from_date, number_of_data) #tick data not high low close bars

    # Transform array into a DataFrame
    df_ticks = pd.DataFrame(ticks)

    # Convert number format of the date into date format
    df_ticks["time"] = pd.to_datetime(df_ticks["time"], unit="s")

    df_ticks = df_ticks.set_index("time")

    return df_ticks

def get_alphaVantage(symbol, interval, key):
    

    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo'
    r = requests.get(url)
    data = r.json()

    print(data)

currency = "EURUSD!"
timeframe = mt5.TIMEFRAME_M5
# !! You can't import more than 99.999 rows in one request
df = get_rates(currency, number_of_data=99_999, timeframe=timeframe) #AUDUSD-Z
#df = get_rates("EURUSD!", number_of_data=99_999, timeframe=mt5.TIMEFRAME_H1)
#df = get_rates("AUDUSD!", number_of_data=99_999, timeframe=mt5.TIMEFRAME_H1)
#df = get_rates("USDJPY", number_of_data=99_999, timeframe=mt5.TIMEFRAME_H4)

#put this as your input FixTimeBars/USDJPY_1H.csv

# Display the data
print(df)

# Put where you want to save the database
#sample
#C:\ws\Alpha-Quant-Course\Data\FixTimeBars\USDCHF_M30_Forex.csv

save_path = get_save_path(currency, timeframe)
df.to_csv(save_path)

# Save the database if you had put a path
'''
save_path = input("Write the path to store the file if you want to (if not, just press enter):")

if len(save_path)>0:
    df.to_csv(save_path)

'''


###### Exercise
#- Do the same thing, for one of the 3 other function (copy_rates_range, copy_ticks_from or copy_ticks_range)
