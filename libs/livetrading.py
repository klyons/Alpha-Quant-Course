import pdb, sys, os
# Get the directory two levels up from the current script's directory
execution_dir = os.path.abspath(os.getcwd())

# Add that directory to the system path
sys.path.append(execution_dir)
from trading import quotes2
from trading import positions
from trading import tradingqueue as tq

class LiveOrder():
		self.symbol = str()
		self.instruction = str() # BUY or SELL
		self.price = 0
		self.bar_size = 0
		self.stop_loss = 0 # in actual price
		self.profit_tgt = 0
		self.quantity = 0
		self.hash = 0
		self.strategy_name = 0

class LiveTrading():
	def __init__(self, broker="schwab"):
		self.log = self.init_log()
		self.broker_name = broker
		# create a connection to the broker using the APIs in
		self.qt = quotes2.Quotes(self.broker_name, log=self.log)
		self.account_hash = self.qt.account_hash
		self.broker = self.qt.broker
		self.positions = positions.Positions(self.broker, self.log)

		self.que = tq.TradingQueue(self.log, callback=None)
		self.host_name = 'localhost'
		self.queue_name = 'equities'
		self.que.connect(host=host_name)
		self.que.create_queue(queue_name)

	def init_log():
		al = bot.AppLog()
		if not os.path.isdir("logs"):
			os.mkdir("logs")	
		log = al.create_log("logs/quantreo_trading.log")
		now = datetime.now()
		formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
		log.info("Start of quantreo_trading log: %s"%formatted_now)
		return log

	def get_open_position(self, symbol):
		status, pos = self.positions.get_open_position(self.qt.broker, symbol, self.account_hash)
		return status, pos

	def get_quote(self, symbol, lookback_days=10):
		df = self.qt.get_broker_quote(symbol, lookback_days=lookback_days, pacific_time=True, regular_trading_time=True)
		if df.empty:
			print("Unable to get data for %s"%symbol)
		return df

	def get_single_quote(self, symbol):
		return self.qt.get_single_quote(symbol)

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

	def exit_position(self, symbol, instruction, quantity, strategy_name):
		open_pos, pos = self.positions.get_open_position(self.qt.broker, symbol, self.account_hash)
		if open_pos:
			market_order = {'order_type': 'MARKET', 'instruction': instruction, 'symbol': symbol,
			 'price': 0, 'stop_loss': 0, 'profit_target': 0.00,
			 'quantity': quantity, 'order_hash': '', 'strategy_name': strategy_name}
			json_message = json.dumps(market_order)
			self.que.send_msg(self.queue_name, json_message)
			self.log.info(market_order)

	def send_order(self, order):
		instruction = order.instruction
		symbol = order.symbol
		price = order.price`
		stop_loss = order.stop_loss
		profit_target = order.profit_tgt
		oco_order = {'order_type': 'LIMIT', 'instruction': instruction, 'symbol': symbol, 'price': price,
		 'stop_loss': stop_loss, 'profit_target': profit_target, 'quantity': order.quantity, 
		 'order_hash': order.hash, 'strategy_name': order.strategy_name}
		json_message = json.dumps(oco_order)
		# send the message and exit
		self.que.send_msg(queue_name, json_message)
		self.log.info(oco_order)
		return True

if __name__ == '__main__':
	data = DataFeed()
	df = data.get_quote('spy', lookback_days=10)
	rdf = data.get_time_bars(df, '60T')
	print(rdf)
	pos = data.get_open_position('FSK')