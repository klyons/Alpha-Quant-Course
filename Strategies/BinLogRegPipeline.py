"""
Description:  Trading strategy based on a Machine Learning algorithm (Logistic Regression). As input, we have
              SMA diff, RSI and ATR to have more information.

              We standardize the data to put all the data at the same scale (necessary for many algorithms and
              allow algorithms to have a better convergence)

Entry signal: We need that the ML algo say to buy in the same time

Exit signal:  Basic Take-profit and Stop-loss

Good to know: Only one trade at time (we can't have a buy and a sell position in the same time)

How to improve this algorithm?: Try a non-linear model to see the difference of performances
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

#for importing the quantreo library
import sys, pdb
sys.path.insert(0, '..')
from Quantreo.DataPreprocessing import *

class BinLogRegPipeline():

    def __init__(self, data, parameters):
        # Set parameters
        self.list_X = parameters["list_X"]
        self.tp, self.sl = parameters["tp"], parameters["sl"]
        self.cost, self.leverage = parameters["cost"], parameters["leverage"]
        self.train_mode = parameters["train_mode"]
        self.sma_fast, self.sma_slow = parameters["sma_fast"], parameters["sma_slow"]
        self.rsi_period, self.atr_period = parameters["rsi"], parameters["atr"]
        self.look_ahead_period = parameters["look_ahead_period"]
        self.lags = parameters["lags"]
        self.multiply_data = False
        if parameters.get("multiply_data"):
            self.multiply_data = parameters["multiply_data"]
        self.columns_added = False

        self.model, self.saved_model_path = None, None

        # Get test parameters
        self.output_dictionary = parameters.copy()
        #change to True for 
        self.output_dictionary["train_mode"] = False

        if self.train_mode:
            self.data_train = data
            self.data = data
            self.train_model()
        else:
            self.model = parameters["model"]
            self.data = data

        self.start_date_backtest = self.data.index[0]
        self.get_predictions()

        # Get Entry parameters
        self.buy, self.sell = False, False
        self.open_buy_price, self.open_sell_price = None, None
        self.entry_time, self.exit_time = None, None

        # Get exit parameters
        self.var_buy_high, self.var_sell_high = None, None
        self.var_buy_low, self.var_sell_low = None, None

    def get_features(self, data_sample):
        data_sample = sma_diff(data_sample, "close", self.sma_fast, self.sma_slow)
        data_sample = rsi(data_sample, "close", self.rsi_period)
        data_sample = candle_information(data_sample)
        data_sample = previous_ret(data_sample, "close", 1)

        data_sample = atr(data_sample,self.atr_period)
        new_columns = []
        for col in self.list_X:
            for lag in range(1, self.lags + 1):
                lagged_col_name = f"{col}_l{lag}"
                if not self.columns_added:
                    data_sample[lagged_col_name] = data_sample[col].shift(lag)
                    new_columns.append(lagged_col_name)
                else:
                    data_sample[lagged_col_name] = data_sample[col].shift(lag)
        if not self.columns_added:
            self.list_X = self.list_X + new_columns
            self.columns_added = True  # Set the flag to True after adding 
        # If there are still NaN values, replace with 0
        data_sample = data_sample.fillna(0)
        return data_sample

    def train_model(self):
        # Create the features and the target
        full_split = 1.00

        # features to reply in the get_features function
        self.data_train = self.get_features(self.data_train)
        # Create lists with the columns name of the features used and the target
        #self.data_train = binary_signal(self.data_train, self.look_ahead_period)
        self.data_train = get_barriers_signal(self.data_train)
        pdb.set_trace()
        self.data_train = self.data_train.dropna()
        # figure out what is going on with the dataframe
        # make sure all the features get returned out of 
        list_y = ["Signal"]

        # Split our dataset in a train and a test set
        split = int(len(self.data_train) * full_split)
        X_train, X_test, y_train, y_test = data_split(self.data_train, split, self.list_X, list_y)
        
        #change cross validation to work with time series
        tscv = TimeSeriesSplit(n_splits=2)

        # Create the model
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(solver='saga', tol=1e-2, max_iter=100, random_state=42))
        ])

        grid = {
            'logistic__C': [1e-3, 1e-1, 1e0, 1e1, 1e2],
            'logistic__penalty': ['l1']  # consider testing over l2 as well
        }
        #add time series cross validation
        ml_model = GridSearchCV(pipe, grid, cv=tscv, n_jobs=-1)
        ml_model.fit(X_train, y_train)

        # print model logistics and 
        # Save models as attributes
        self.model = ml_model.best_estimator_

        # Save the model for the eventual test sets
        self.output_dictionary["model"] = ml_model

    def get_predictions(self):
        self.data = self.get_features(self.data)

        X = self.data[self.list_X]

        predict_array = self.model.predict(X)
        self.data["ml_signal"] = 0
        self.data["ml_signal"] = predict_array

    def get_entry_signal(self, time):
        """
        Random Entry signal
        :param i: row number
        :return: Open a buy or sell position using a random signal
        """
        if self.data.index.name == 'date_time':
            self.data.rename_axis('time', inplace=True)
            
        if time not in self.data.index:
            return 0, self.entry_time

        if len(self.data.loc[:time]["ml_signal"]) < 2:
            return 0, self.entry_time

        # Create entry signal --> -1,0,1
        entry_signal = 0
        if self.data.loc[:time]["ml_signal"][-2] == 1:
            entry_signal = 1
        elif self.data.loc[:time]["ml_signal"][-2] == -1:
            entry_signal = -1

        # Enter in buy position only if we want to, and we aren't already
        if entry_signal == 1 and not self.buy and not self.sell:
            self.buy = True
            self.open_buy_price = self.data.loc[time]["open"]
            self.entry_time = time

        # Enter in sell position only if we want to, and we aren't already
        elif entry_signal == -1 and not self.sell and not self.buy:
            self.sell = True
            self.open_sell_price = self.data.loc[time]["open"]
            self.entry_time = time

        else:
            entry_signal = 0

        return entry_signal, self.entry_time

    def get_exit_signal(self, time):
        """
        Take-profit & Stop-loss exit signal
        :param i: row number
        :return: P&L of the position IF we close it

        **ATTENTION**: If you allow your bot to take a buy and a sell position in the same time,
        you need to return 2 values position_return_buy AND position_return_sell
        """
        # Verify if we need to close a position and update the variations IF we are in a buy position
        if self.buy:
            self.var_buy_high = (self.data.loc[time]["high"] - self.open_buy_price) / self.open_buy_price
            self.var_buy_low = (self.data.loc[time]["low"] - self.open_buy_price) / self.open_buy_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_buy_high) and (self.var_buy_low < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_buy, self.exit_time

                else:
                    self.buy = False
                    self.open_buy_price = None
                    position_return_buy = 0
                    self.exit_time = time
                    return position_return_buy, self.exit_time

            elif self.tp < self.var_buy_high:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_buy_low < self.sl:
                self.buy = False
                self.open_buy_price = None
                position_return_buy = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_buy, self.exit_time

        # Verify if we need to close a position and update the variations IF we are in a sell position
        if self.sell:
            self.var_sell_high = -(self.data.loc[time]["high"] - self.open_sell_price) / self.open_sell_price
            self.var_sell_low = -(self.data.loc[time]["low"] - self.open_sell_price) / self.open_sell_price

            # Let's check if AT LEAST one of our threshold are touched on this row
            if (self.tp < self.var_sell_low) and (self.var_sell_high < self.sl):

                # Close with a positive P&L if high_time is before low_time
                if self.data.loc[time]["low_time"] < self.data.loc[time]["high_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.tp - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                # Close with a negative P&L if low_time is before high_time
                elif self.data.loc[time]["high_time"] < self.data.loc[time]["low_time"]:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = (self.sl - self.cost) * self.leverage
                    self.exit_time = time
                    return position_return_sell, self.exit_time

                else:
                    self.sell = False
                    self.open_sell_price = None
                    position_return_sell = 0
                    self.exit_time = time
                    return position_return_sell, self.exit_time

            # Close with a positive P&L if high_time is before low_time
            elif self.tp < self.var_sell_low:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.tp - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

            # Close with a negative P&L if low_time is before high_time
            elif self.var_sell_high < self.sl:
                self.sell = False
                self.open_sell_price = None
                position_return_sell = (self.sl - self.cost) * self.leverage
                self.exit_time = time
                return position_return_sell, self.exit_time

        return 0, None