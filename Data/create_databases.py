from datetime import datetime, timedelta, date
import pandas as pd
import MetaTrader5 as mt5
import requests
from dotenv import load_dotenv
import os
import sys
import pdb

from lib import databank
from lib import utils

class DataHandler:
    def __init__(self):
        load_dotenv()
        self.polygon_api_key = os.getenv('polygon_api_key')
        self.polygon_rest_baseurl = "https://api.polygon.io/v2/"
        self.limit = 100000

    def get_equities_save_path(self, symbol, multiplier, timespan):
        timespan_map = {
            "day": "D",
            "hour": "H",
            "minute": "M",
            "second": "S"
        }
        t = timespan_map.get(timespan, "")
        cwd = os.getcwd()
        return os.path.join(cwd, f"quantreo/Data/Equities/{multiplier}{t}/{symbol}_{multiplier}{t}.parquet")

    def get_save_path(self, symbol, timeframe):
        symbol = symbol.replace("!", "_")
        timeframe_dict = {
            mt5.TIMEFRAME_M1: "1M",
            mt5.TIMEFRAME_M5: "5M",
            mt5.TIMEFRAME_M10: "10M",
            mt5.TIMEFRAME_M15: "15M",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "1H",
            mt5.TIMEFRAME_H4: "4H"
        }
        timeframe_str = timeframe_dict.get(timeframe, "Unknown")
        save_path = f"C:/ws/copernicus/quantreo/Data/Currencies/{timeframe_str}/{symbol}{timeframe_str}.parquet"
        return save_path

    def get_rates(self, symbol, number_of_data=10_000, timeframe=mt5.TIMEFRAME_D1):
        from_date = datetime.now()
        rates = mt5.copy_rates_from(symbol, timeframe, from_date, number_of_data)
        df_rates = pd.DataFrame(rates)
        df_rates["time"] = pd.to_datetime(df_rates["time"], unit="s")
        df_rates = df_rates.set_index("time")
        return df_rates

    def get_ticks(self, symbol, number_of_data=10_000, timeframe=mt5.TIMEFRAME_D1):
        from_date = datetime.now()
        ticks = mt5.copy_ticks_from(symbol, timeframe, from_date, number_of_data)
        df_ticks = pd.DataFrame(ticks)
        df_ticks["time"] = pd.to_datetime(df_ticks["time"], unit="s")
        df_ticks = df_ticks.set_index("time")

    def get_tickers(self):
        request_url = f"https://api.polygon.io/v3/reference/tickers?active=true&limit=1000&apiKey={self.polygon_api_key}"
        data = requests.get(request_url).json()
        if data['status'] == 'OK':
            df = pd.DataFrame(data['results'])
            next_d = data['next_url']
            while(next_d != {}):
                next_d = f"{next_d}&apiKey={self.polygon_api_key}"
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

    def pull_data(self, symbol, multiplier, timespan, start_time = None):  #added = None
        sort = "asc"
        if not start_time:
            start_time = datetime.today() - timedelta(days=365*5)
            start_time = datetime.date(start_time)
        end_time = date.today()

        request_url2 = f"{self.polygon_rest_baseurl}aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_time}/{end_time}?adjusted=true&sort={sort}&limit={self.limit}&apiKey={self.polygon_api_key}"

        if symbol[:2] == 'I:' or symbol[:2] == 'i:':
            request_url = f"{self.polygon_rest_baseurl}aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_time}/{end_time}?adjusted=true&sort={sort}&limit={self.limit}&apiKey={self.polygon_api_key}"
        else:
            request_url = f"{self.polygon_rest_baseurl}aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_time}/{end_time}?adjusted=true&sort={sort}&limit={self.limit}&apiKey={self.polygon_api_key}"

        try:
            data = requests.get(request_url).json()
        except:
            print("Connection Error, try again")
            return None

        if "results" in data:
            return data["results"]
        else:
            return None

    def get_equity(self, symbol, multiplier=1, timespan='hour', silent=False, start_day=None, end_day=None):
        bars = []
        mdf = pd.DataFrame()
        if len(timespan) < 2:
            timespan_map = {
                "D": "day",
                "H": "hour",
                "M": "minute",
                "S": "second"
            }
            timespan = timespan_map.get(timespan, "")
        interval = timespan
        symbol = symbol.upper()
        if not start_day:
            start_day = datetime.today() - timedelta(days=364*5)
            start_day = datetime.date(start_day)
        while(1):
            bars = self.pull_data(symbol, timespan=timespan, start_time=start_day, multiplier=multiplier)
            if not bars:
                if not silent:
                    print(f"No data available for {symbol} in this range {start_day}")
                break
            df = pd.DataFrame(bars)
            df["date_time"] = pd.to_datetime(df["t"], unit="ms")
            if ':' in symbol:
                df = df[["date_time", "o", "h", "c", "l"]]
                df.columns = ["date_time", "open", "high", "close", "low"]
            if not ':' in symbol:
                df = df[["date_time", "o", "h", "c", "l", "v", "vw"]]
                df.columns = ["date_time", "open", "high", "close", "low", "volume", "vwap"]
            df = df.sort_values("date_time")
            if not silent and not df.empty:
                print(f"Downloaded {symbol}: {df['date_time'].iloc[0]} - {df['date_time'].iloc[-1]}")

            if df['date_time'].iloc[0] == df['date_time'].iloc[-1]:
                mdf = df
                break

            df.set_index(['date_time'], inplace=True)
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
        save_path = self.get_equities_save_path(symbol, multiplier, timespan)
        mdf.to_parquet(save_path)

    def get_currency(self, currency=None, timeframe=None):
        if currency is None:
            currency = "EURUSD!"
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M5
        df = self.get_rates(currency, number_of_data=99_999, timeframe=timeframe)
        print(df)

# Example usage
if __name__ == '__main__':
    handler = DataHandler()
    handler.get_equity("AAPL", multiplier=1, timespan='day')
    handler.get_currency("EURUSD!", timeframe=mt5.TIMEFRAME_M5)
