import pdb, sys, os
# Get the directory two levels up from the current script's directory
two_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add that directory to the system path
sys.path.append(two_up)
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