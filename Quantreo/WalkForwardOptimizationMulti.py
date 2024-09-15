import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from Quantreo.Backtest import Backtest

class WalkForwardOptimizationMulti:
    def __init__(self, main_data, additional_data, TradingStrategy, fixed_parameters, parameters_range, 
                 length_train_set=10_000, pct_train_set=.80, anchored=True, title=None, randomness=0.75):
        # Set initial parameters
        self.main_data = main_data
        self.additional_data = additional_data
        self.TradingStrategy = TradingStrategy
        self.parameters_range = parameters_range
        self.fixed_parameters = fixed_parameters
        self.randomness = randomness
        self.dictionaries = None
        self.get_combinations()

        # Necessary variables to create our sub-samples
        self.length_train_set, self.pct_train_set = length_train_set, pct_train_set
        self.train_samples, self.test_samples, self.anchored = [], [], anchored

        # Necessary variables to compute and store our criteria
        self.BT, self.criterion = None, None
        self.best_params_sample_df, self.best_params_sample = None, None
        self.smooth_result = pd.DataFrame()
        self.best_params_smoothed = list()

        # Create dataframe that will contain the optimal parameters  (ranging parameters + criterion)
        self.columns = list(self.parameters_range.keys())
        self.columns.append("criterion")
        self.df_results = pd.DataFrame(columns=self.columns)

        # Set the title of our Backtest plot
        self.title_graph = title

    def get_combinations(self):
        # Create a list of dictionaries with all the possible combination (ONLY with variable parameters)
        keys = list(self.parameters_range.keys())
        combinations = list(itertools.product(*[self.parameters_range[key] for key in keys]))
        self.dictionaries = [dict(zip(keys, combination)) for combination in combinations]

        # We add the fixed parameters on each dictionary
        for dictionary in self.dictionaries:
            dictionary.update(self.fixed_parameters)

    def get_sub_samples(self):
        # Compute the length of the test set
        length_test = int(self.length_train_set / self.pct_train_set - self.length_train_set)

        # Initialize size parameters
        start = 0
        end = self.length_train_set

        # We split the data until we can't make more than 2 sub-samples
        while (len(self.main_data) - end) > 2 * length_test:
            end += length_test

            # If we are at the last sample we take the whole rest to not create a tiny last sample
            if (len(self.main_data) - end) < 2 * length_test:
                if self.anchored:
                    self.train_samples.append((self.main_data.iloc[:end - length_test, :], 
                                               [df.iloc[:end - length_test, :] for df in self.additional_data]))
                    self.test_samples.append((self.main_data.iloc[end - length_test:, :], 
                                              [df.iloc[end - length_test:, :] for df in self.additional_data]))
                else:
                    self.train_samples.append((self.main_data.iloc[start:end - length_test, :], 
                                               [df.iloc[start:end - length_test, :] for df in self.additional_data]))
                    self.test_samples.append((self.main_data.iloc[end - length_test:, :], 
                                              [df.iloc[end - length_test:, :] for df in self.additional_data]))
                break

            if self.anchored:
                self.train_samples.append((self.main_data.iloc[:end - length_test, :], 
                                           [df.iloc[:end - length_test, :] for df in self.additional_data]))
                self.test_samples.append((self.main_data.iloc[end - length_test:end, :], 
                                          [df.iloc[end - length_test:end, :] for df in self.additional_data]))
            else:
                self.train_samples.append((self.main_data.iloc[start:end - length_test, :], 
                                           [df.iloc[start:end - length_test, :] for df in self.additional_data]))
                self.test_samples.append((self.main_data.iloc[end - length_test:end, :], 
                                          [df.iloc[end - length_test:end, :] for df in self.additional_data]))

            start += length_test

    def get_criterion(self, main_sample, additional_samples, params):
        # Backtest initialization with a specific dataset and set of parameters
        self.BT = Backtest(data=main_sample, additional_data=additional_samples, 
                           TradingStrategy=self.TradingStrategy, parameters=params)

        # Compute the returns of the strategy
        self.BT.run()

        # Calculation and storage of the criterion (Return over period over the maximum drawdown)
        ret, dd = self.BT.get_ret_dd()

        # We add ret and dd because dd < 0
        self.criterion = ret + 2*dd

    def get_best_params_train_set(self):
        storage_values_params = []

        for self.params_item in np.random.choice(self.dictionaries, size=int(len(self.dictionaries)*self.randomness), replace=False):
            current_params = [self.params_item[key] for key in list(self.parameters_range.keys())]

            self.get_criterion(self.train_sample[0], self.train_sample[1], self.params_item)
            current_params.append(self.criterion)

            storage_values_params.append(current_params)

        df_find_params = pd.DataFrame(storage_values_params, columns=self.columns)

        self.best_params_sample_df = df_find_params.sort_values(by="criterion", ascending=False).iloc[0:1, :]
        self.best_params_sample_df.index = self.train_sample[0].index[-2:-1]

        self.df_results = pd.concat((self.df_results, self.best_params_sample_df), axis=0)

        self.best_params_sample = dict(df_find_params.sort_values(by="criterion", ascending=False).iloc[0, :-1])
        self.best_params_sample.update(self.fixed_parameters)

    def get_smoother_result(self):
        self.smooth_result = pd.DataFrame()

        for column in self.df_results.columns:
            if isinstance(self.df_results[column][0], (float, np.float64)):
                self.smooth_result[column] = self.df_results[column].ewm(com=1.5, ignore_na=True).mean()
            else:
                self.smooth_result[column] = self.df_results[column].mode()

        test_params = dict(self.smooth_result.iloc[-1,:-1])

        Strategy = self.TradingStrategy(self.train_sample[0], self.best_params_sample, *self.train_sample[1])

        output_params = Strategy.output_dictionary

        for key in test_params.keys():
            output_params[key] = test_params[key]

        return output_params

    def test_best_params(self):
        smooth_best_params = self.get_smoother_result()

        self.get_criterion(self.test_sample[0], self.test_sample[1], smooth_best_params)

        self.df_results.at[self.df_results.index[-1], 'criterion'] = self.criterion
        self.best_params_smoothed.append(smooth_best_params)

    def run_optimization(self):
        self.get_sub_samples()

        for self.train_sample, self.test_sample in tqdm(zip(self.train_samples, self.test_samples)):
            self.get_best_params_train_set()
            self.test_best_params()

    def display(self):
        df_test_result = pd.DataFrame()

        for params, test in zip(self.best_params_smoothed, self.test_samples):
            self.BT = Backtest(data=test[0], additional_data=test[1], 
                               TradingStrategy=self.TradingStrategy, parameters=params)
            self.BT.run()
            df_test_result = pd.concat((df_test_result, self.BT.data), axis=0)

        self.BT = Backtest(data=df_test_result, additional_data=self.additional_data, 
                           TradingStrategy=self.TradingStrategy, parameters=params)
        self.BT.display(self.title_graph)