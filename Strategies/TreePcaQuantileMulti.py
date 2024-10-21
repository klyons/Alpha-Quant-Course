"""
Description:  Trading strategy based on a Machine Learning algorithm (Decision Tree). As input, we take a LOT of
              different features that we will reduce using a PCA

              We standardize the data to put all the data at the same scale (necessary for PCA, not especially for the
              Decision Tree)

              We apply an PCA to reduce the number of variable and remove the multicolinearity


Entry signal: We need that the ML algo say to buy in the same time

Exit signal:  Basic Take-profit and Stop-loss

Good to know: Only one trade at time (we can't have a buy and a sell position in the same time)

How to improve this algorithm?: Put variable Take-profit and Stop loss
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pdb, sys, os

# Get the current working directory
current_working_directory = os.getcwd()
# Construct the path to the quantreo folder
quantreo_path = os.path.join(current_working_directory, 'quantreo')
# Add the quantreo folder to the Python path
sys.path.append(quantreo_path)

from Quantreo.DataPreprocessing import *
from Strategies.Strategy import Strategy
#for importing the quantreo library


class TreePcaQuantileMulti():  #try this without inheriting from Strategy first
    def __init__(self, data, parameters, **kwargs):
        #super().__init__(data, parameters, *kwargs)
        # Set parameters
        self.list_X = parameters["list_X"]
        self.tp, self.sl = parameters["tp"], parameters["sl"]
        self.cost, self.leverage = parameters["cost"], parameters["leverage"]
        self.train_mode = parameters["train_mode"]
        self.sma_fast, self.sma_slow = parameters["sma_fast"], parameters["sma_slow"]
        self.rsi_period, self.atr_period = parameters["rsi"], parameters["atr"]
        self.look_ahead_period = parameters["look_ahead_period"]
        self.lags = parameters["lags"]
        self.dataframes = kwargs

        self.model, self.sc, self.pca = None, None, None
        self.saved_model_path, self.saved_sc_path = None, None

        # Get test parameters
        self.output_dictionary = parameters.copy()
        self.output_dictionary["train_mode"] = False

        if self.train_mode:
            self.data_train = data
            self.data = data
            self.train_model()
        else:
            self.model = parameters["model"]
            self.sc = parameters["sc"]
            self.pca = parameters["pca"]
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

        # Process additional DataFrames
        

    def get_features(self, data_sample, symbol=None):
        data_sample = sma_diff(data_sample, "close", self.sma_fast, self.sma_slow)
        data_sample = rsi(data_sample, "close", self.rsi_period)
        data_sample = previous_ret(data_sample, "close", 60)
        data_sample = sto_rsi(data_sample, "close", 14)
        data_sample = ichimoku(data_sample, 27, 78)
        data_sample = candle_information(data_sample)
        data_sample = atr(data_sample, self.atr_period)
        new_columns = []
        #add multiplier features
        data_sample = self.add_multiplier_features(data_sample)
        
        # Add lag features
        data_sample = self.add_lag_features(data_sample)
        #fill na values with 0
        data_sample = data_sample.fillna(value=0)
        return data_sample

    def train_model(self):
        # Create the features and the target
        full_split = 1.00

        # features to reply in the get_features function
        self.data_train = self.get_features(self.data_train)
        # need to add the symbols to the columns of additional data. 
        self.additional_data = [self.get_features(df, symbol=symbol_name) for symbol_name, df in self.dataframes.items()]
        #concatinate self.additional_data and data_train
        self.data_train = pd.concat([self.data_train] + self.additional_data, axis=1)
        # Create lists with the columns name of the features used and the target
        self.data_train = quantile_signal(self.data_train, self.look_ahead_period, pct_split=full_split)
        # !! As it is very time-consuming to compute & it is not variable, we compute it outside the function
            
        list_y = ["Signal"]

        # Split our dataset in a train and a test set
        split = int(len(self.data_train) * full_split)
        X_train, X_test, y_train, y_test = data_split(self.data_train, split, self.list_X, list_y)

        # Initialize the standardization model
        #sc = StandardScaler()
        #X_train_sc = sc.fit_transform(X_train)

        # Create a PCA to remove multicolinearity and reduce the number of variable keeping many information
        #pca = PCA(n_components=3)
        #X_train_pca = pca.fit_transform(X_train_sc)
        tscv = TimeSeriesSplit(n_splits=2)
        # Create the model
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('pca', PCA(n_components=5)),
            ('clf', DecisionTreeClassifier())
        ])

        # Define the hyperparameters to search over
        grid = {
            'pca__n_components': [15, 20, 25], #len(df.columns // 1.2)
            'clf__min_samples_split': [5, 10],
            'clf__max_depth': [4, 6, 8]
        }

        
        ml_model = GridSearchCV(pipe, grid, cv=tscv)
        ml_model.fit(X_train, y_train)
        
        # Save models as attributes
        self.model = ml_model.best_estimator_
        self.sc = ml_model.best_estimator_.named_steps['sc']
        self.pca = ml_model.best_estimator_.named_steps['pca']

        self.output_dictionary["model"] = ml_model
        self.output_dictionary["sc"] = ml_model.best_estimator_.named_steps['sc']
        self.output_dictionary["pca"] = ml_model.best_estimator_.named_steps['pca']

    def get_predictions(self):
        #make sure all the data makes it into this part 
        #else add 
        # need to add the symbols to the columns of additional data. 
        self.additional_data = [self.get_features(df, symbol=symbol_name) for symbol_name, df in self.dataframes.items()]
        #concatinate self.additional_data and data_train
        self.data = self.get_features(self.data)
        
        self.data = pd.concat([self.data] + self.additional_data, axis=1)
        
        pdb.set_trace()
        X = self.data[self.list_X]
        #X_sc = self.sc.transform(X)
        #X_pca = self.pca.transform(X_sc)
        #original code
        #predict_array = self.model.predict(X_pca)
        predict_array = self.model.predict(X)
        self.data["ml_signal"] = 0
        self.data["ml_signal"] = predict_array

    def get_entry_signal(self, time):
        """
        Random Entry signal
        :param i: row number
        :return: Open a buy or sell position using a random signal
        """
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
