from datetime import datetime
import pandas as pd
import MetaTrader5 as mt5
import requests
from dotenv import load_dotenv
import os
import sys
import pdb


mt5.initialize()
print(mt5.account_info())
load_dotenv()
polygon_api_key = os.getenv('polygon_api_key')


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
import requests
import pandas as pd
import pdb
import time
from datetime import date, datetime, timedelta

#https://polygon.io/docs/stocks/getting-started

polygon_api_key = "JlgRBshWYNKUeDMpSLOjvk7slj1GCpBh"  #JlgRBshWYNKUeDMpSLOjvk7slj1GCpBh
polygon_rest_baseurl = "https://api.polygon.io/v2/"  #maybe trie v3as well

limit = 100000

def get_tickers():
    request_url = "https://api.polygon.io/v3/reference/tickers?active=true&limit=1000&apiKey={0}".format(polygon_api_key)
    data = requests.get(request_url).json()
    if data['status'] == 'OK':
        df = pd.DataFrame(data['results'])
        next_d = data['next_url']
        while(next_d != {}):
            next_d = "%s&apiKey=%s"%(next_d, polygon_api_key)
            data = requests.get(next_d).json()
            if data['status'] == 'OK':
                tdf = pd.DataFrame(data['results'])
                df = pd.concat([df, tdf])
            if "next_url" in data.keys():
                next_d = data['next_url']
            else:
                break
        df.to_csv("tickers.csv")
        print("Dumped all tickers to tickers.csv")
    else:
        print("No tickers downloaded")

def pull_data(symbol, start_time, multiplier, timespan):
    """
    date is a python date format
    symbol is of the form XXXUSD
    """

    # newest data at the bottom
    sort = "asc"

    if not start_time:
        start_time = datetime.today() - timedelta(days=365*5) # polygon has a 5 year limitation
        start_time =  datetime.date(start_time)
    end_time = today = date.today()

    ## this means the symbol is an index and requires a differant API call

    request_url2 = f"{polygon_rest_baseurl}aggs/ticker/{symbol}/range/{multiplier}/" +\
            f"{timespan}/{start_time}/{end_time}?adjusted=true&sort={sort}&" + \
            f"limit={limit}&apiKey={polygon_api_key}"

    if symbol[:2] == 'I:' or symbol[:2] == 'i:':  ## add string for index or symbol to allow for adjustment
        request_url = "{0}aggs/ticker/{1}/range/{2}/{3}/{4}/{5}?adjusted=true&sort={6}&limit={7}&apiKey={8}".format(polygon_rest_baseurl, symbol, multiplier, timespan, start_time, end_time, sort, limit, polygon_api_key)

    else:
        request_url = "{0}aggs/ticker/{1}/range/{2}/{3}/{4}/{5}?adjusted=true&sort={6}&limit={7}&apiKey={8}".format(polygon_rest_baseurl, symbol, multiplier, timespan, start_time, end_time, sort, limit, polygon_api_key)


    # If there is a connection error, the following try and catch should prevent Python from
    # exiting. Instead, return no data and let the caller try again
    try:
        data = requests.get(request_url).json()
    except:
        print("Connection Error, try again")
        return None 

    if "results" in data:
        return data["results"]
    else:
        return None


def pull_dividend_data(symbol, start_time, multiplier, timespan):
    """
    date is a python date format
    symbol is of the form XXXUSD
    """

    # newest data at the bottom
    sort = "asc"

    if not start_time:
        start_time = datetime.today() - timedelta(days=365*5) # polygon has a 5 year limitation
        start_time =  datetime.date(start_time)
    end_time = today = date.today()

    ## this means the symbol is an index and requires a differant API call
    url = f'https://api.polygon.io/v3/reference/dividends/{symbol}?from={start_date}&to={end_date}&apiKey={polygon_api_key}'


    request_url2 = f"{polygon_rest_baseurl}aggs/ticker/{symbol}/range/{multiplier}/" +\
            f"{timespan}/{start_time}/{end_time}?adjusted=true&sort={sort}&" + \
            f"limit={limit}&apiKey={polygon_api_key}"

    if symbol[:2] == 'I:' or symbol[:2] == 'i:':  ## add string for index or symbol to allow for adjustment
        request_url = "{0}aggs/ticker/{1}/range/{2}/{3}/{4}/{5}?adjusted=true&sort={6}&limit={7}&apiKey={8}".format(polygon_rest_baseurl, symbol, multiplier, timespan, start_time, end_time, sort, limit, polygon_api_key)

    else:
        request_url = "{0}aggs/ticker/{1}/range/{2}/{3}/{4}/{5}?adjusted=true&sort={6}&limit={7}&apiKey={8}".format(polygon_rest_baseurl, symbol, multiplier, timespan, start_time, end_time, sort, limit, polygon_api_key)


    # If there is a connection error, the following try and catch should prevent Python from
    # exiting. Instead, return no data and let the caller try again
    try:
        data = requests.get(request_url).json()
    except:
        print("Connection Error, try again")
        return None 

    if "results" in data:
        return data["results"]
    else:
        return None


def get_data(start_day, symbol, multiplier=1, timespan='hour', silent=False):
    # start_day must be of datetime.date() format
    # symbol must be uppercase
    bars = []
    mdf = pd.DataFrame()
    interval = timespan
    symbol = symbol.upper()
    while(1):
        bars = pull_data(symbol, start_day, multiplier, interval)
        if not bars:
            if not silent:
                print("No data available for %s in this range %s "%(symbol, start_day))
            break
        df = pd.DataFrame(bars)
        df["date_time"] = pd.to_datetime(df["t"], unit = "ms")
        if ':' in symbol:
            df = df[["date_time","o","h","c","l"]]
            df.columns = ["date_time","open","high","low","close"]
        if not ':' in symbol:    
            df =  df[["date_time","o","h","c","l","v","vw"]]
            df.columns = ["date_time","open","high","low","close","volume","vwap"]
        df = df.sort_values("date_time")
        if not silent and not df.empty:
            print("Downloaded %s: %s - %s"%(symbol, df['date_time'].iloc[0], df['date_time'].iloc[-1]))

        # When a symbol does not have anymore data, check for the start and end date here
        # and if they are the same, then exit
        if df['date_time'].iloc[0] == df['date_time'].iloc[-1]:
            mdf = df
            break

        df.set_index(['date_time'], inplace=True)
        # Make sure to get all of the data to this date
        td = pd.to_datetime(date.today()) - df.index[-1]
        if td.days > 1 or td.days == -1:
            start_day = bars[-1]['t']
            if interval == 'hour':
                start_day = start_day + 60*60*1000*multiplier
            elif interval == 'minute':
                start_day = start_day + 60*1000*multiplier
            elif interval == 'day':
                start_day = start_day + 24*60*60*1000
            else:
                print("error, please pass in correct string for time granularity")
            if mdf.empty:
                mdf = df
            else:
                mdf = pd.concat([mdf, df], axis=0)
        else:
            if mdf.empty:
                mdf = df
                break
            else:
                mdf = pd.concat([mdf, df], axis=0)
                break
    #return mdf
    save_path = get_save_path(symbol, interval)
    mdf.to_csv(save_path)


def get_currency():
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


if __name__ == '__main__':
    #get_currency()
    get_tickers()