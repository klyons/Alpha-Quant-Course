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

	def get_quote(self, symbol):
		df = self.qt.get_broker_quote(symbol, lookback_days=10)
		if df.empty:
			print("Unable to get data for %s"%symbol)
		return df

	def example_quote(self):
		df = self.get_quote("spy")
		print(df)
if __name__ == '__main__':
	data = DataFeed()
	data.example_quote()