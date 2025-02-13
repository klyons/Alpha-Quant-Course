import pdb, sys, os
# Get the directory two levels up from the current script's directory
execution_dir = os.path.abspath(os.getcwd())

# Add that directory to the system path
sys.path.append(execution_dir)
from trading import quotes2

class DataFeed():
	def __init__(self, broker="schwab"):
		self.broker = broker
		# create a connection to the broker using the APIs in
		self.qt = quotes2.Quotes(self.broker, log=None)

	def get_quote(self, symbol, lookback_days=10):
		df = self.qt.get_broker_quote(symbol, lookback_days=lookback_days, pacific_time=True, regular_trading_time=True)
		if df.empty:
			print("Unable to get data for %s"%symbol)
		return df

	def get_time_bars(self, df, frequency):
		if df.empty:
			print("Dataframe is empty, unable to get time bars")
			return df
		# convert the dataframe into time bars
		# frequency: 3T, 5T, or 20T or 60T, etc
		# Resample the DataFrame to frequency-minute intervals
		rdf = df.resample(frequency).agg({
		    'open': 'first',
		    'high': 'max',
		    'low': 'min',
		    'close': 'last',
		    'volume': 'sum'
		})
		rdf.dropna(inplace=True)
		return rdf

	def example_quote(self):
		df = self.get_quote("spy")
		print(df)
if __name__ == '__main__':
	data = DataFeed()
	df = data.get_quote('spy', lookback_days=10)
	rdf = data.get_time_bars(df, '60T')
	print(rdf)