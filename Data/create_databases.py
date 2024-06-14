from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5
import sys
sys.path.insert(0, '..')
# Initialize the bounds between MetaTrader 5 and Python
mt5.initialize()


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

# !! You can't import more than 99.999 rows in one request
df = get_rates("USDCHF", number_of_data=99_999, timeframe=mt5.TIMEFRAME_M10) #AUDUSD-Z
#df = get_rates("USDJPY", number_of_data=99_999, timeframe=mt5.TIMEFRAME_H1)
#df = get_rates("USDJPY", number_of_data=99_999, timeframe=mt5.TIMEFRAME_H4)

# Display the data
print(df)

# Put where you want to save the database
#sample
#C:\ws\Alpha-Quant-Course\Data\FixTimeBars\USDCHF_M10_Admiral.csv
save_path = input("Write the path to store the file if you want to (if not, just press enter):")

# Save the database if you had put a path
if len(save_path)>0:
    df.to_csv(save_path)



###### Exercise
#- Do the same thing, for one of the 3 other function (copy_rates_range, copy_ticks_from or copy_ticks_range)
