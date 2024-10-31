from cv2 import add
from torch import empty
from tqdm import tqdm
from datetime import datetime
from termcolor import colored
from Quantreo.Backtest import Backtest
import pdb
import itertools
import pandas as pd
import numpy as np

class WalkForwardOptimizationMulti:
    def __init__(self, main_data, TradingStrategy, fixed_parameters, parameters_range, 
                 length_train_set=10_000, pct_train_set=.80, anchored=True, title=None, randomness=0.75, **additional_data):
        # Set initial parameters
        self.main_data = main_data
        self.TradingStrategy = TradingStrategy
        self.parameters_range = parameters_range
        self.fixed_parameters = fixed_parameters
        self.randomness = randomness
        self.dictionaries = None
        self.get_combinations()
        self.additional_data = additional_data

        # Necessary variables to create our sub-samples
        self.length_train_set, self.pct_train_set = length_train_set, pct_train_set
        self.train_samples, self.test_samples = [], []
        self.add_train_samples, self.add_test_samples = [], []
        self.anchored = anchored

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

        # Align the indices of the additional dataframes with the main data
        aligned_additional_data = {key: df.reindex(self.main_data.index) for key, df in self.additional_data.items()}

        # We split the data until we can't make more than 2 sub-samples
        while (len(self.main_data) - end) > 2 * length_test:
            end += length_test

            # Determine if we are at the last sample
            is_last_sample = (len(self.main_data) - end) < 2 * length_test

            # Determine the slices for training and testing data
            if self.anchored:
                train_slice = self.main_data.iloc[:end - length_test, :]
                test_slice = self.main_data.iloc[end - length_test:, :] if is_last_sample else self.main_data.iloc[end - length_test:end, :]
            else:
                train_slice = self.main_data.iloc[start:end - length_test, :]
                test_slice = self.main_data.iloc[end - length_test:, :] if is_last_sample else self.main_data.iloc[end - length_test:end, :]

            # Slice additional dataframes in aligned_additional_data
            train_additional = {key: df.loc[train_slice.index] for key, df in aligned_additional_data.items()}
            test_additional = {key: df.loc[test_slice.index] for key, df in aligned_additional_data.items()}

            # Append the slices to the samples
            self.train_samples.append(train_slice)
            self.test_samples.append(test_slice)
            self.add_train_samples.append(train_additional)
            self.add_test_samples.append(test_additional)

                # Add assertions to check the shapes
            for key, df in train_additional.items():
                assert df.shape[0] == train_slice.shape[0], f"Shape mismatch in train_additional for key {key}: {df.shape[0]} != {train_slice.shape[0]}"
            for key, df in test_additional.items():
                assert df.shape[0] == test_slice.shape[0], f"Shape mismatch in test_additional for key {key}: {df.shape[0]} != {test_slice.shape[0]}"

            # Break if it's the last sample
            if is_last_sample:
                break

            start += length_test

    def get_criterion(self, sample, params, additional_data):
        # Backtest initialization with a specific dataset and set of parameters
        
        self.BT = Backtest(data=sample, TradingStrategy=self.TradingStrategy, parameters=params, **additional_data)

        # Compute the returns of the strategy
        self.BT.run()

        # Calculation and storage of the criterion (Return over period over the maximum drawdown)
        ret, dd = self.BT.get_ret_dd()

        # We add ret and dd because dd < 0
        self.criterion = ret + 2 * dd

    def get_best_params_train_set(self):
        storage_values_params = []

        for self.params_item in np.random.choice(self.dictionaries, size=int(len(self.dictionaries) * self.randomness), replace=False):
            current_params = [self.params_item[key] for key in list(self.parameters_range.keys())]
            
            # Use the current index to access the correct additional data
            #additional_data = self.add_train_sample

            # Pass train_additional as additional_data to get_criterion
            self.get_criterion(self.train_sample, self.params_item, self.add_train_sample)
            current_params.append(self.criterion)

            storage_values_params.append(current_params)

        df_find_params = pd.DataFrame(storage_values_params, columns=self.columns)

        self.best_params_sample_df = df_find_params.sort_values(by="criterion", ascending=False).iloc[0:1, :]
        self.best_params_sample_df.index = self.train_sample.index[-2:-1]

        self.df_results = pd.concat((self.df_results, self.best_params_sample_df), axis=0)

        self.best_params_sample = dict(df_find_params.sort_values(by="criterion", ascending=False).iloc[0, :-1])
        self.best_params_sample.update(self.fixed_parameters)

    def process_samples(self):
        for train_sample, add_train_sample in zip(self.train_samples, self.add_train_samples):
            self.train_sample = train_sample
            self.add_train_sample = add_train_sample
            self.get_best_params_train_set()

        for test_sample, add_test_sample in zip(self.test_samples, self.add_test_samples):
            self.test_sample = test_sample
            self.add_test_sample = add_test_sample
            # You can call a similar method for testing if needed

    def get_smoother_result(self):
        self.smooth_result = pd.DataFrame()

        for column in self.df_results.columns:
            if isinstance(self.df_results[column][0], (float, np.float64)):
                self.smooth_result[column] = self.df_results[column].ewm(com=1.5, ignore_na=True).mean()
            else:
                self.smooth_result[column] = self.df_results[column].mode()

        test_params = dict(self.smooth_result.iloc[-1,:-1])

        add_train_data = {key: df.loc[self.train_sample.index] for key, df in self.additional_data.items()}
        
        first_key, first_value = next(iter(add_train_data.items()))

        # Print debugging information
        print(f"First key in add_data: {first_key}")
        print(f"First value in add_data: {first_value}")

        # Check if the indices of the two DataFrames match
        if not self.train_sample.index.equals(first_value.index):
            raise ValueError("Indices of train_sample and add_train_data do not match.")

            
        Strategy = self.TradingStrategy(self.train_sample, self.best_params_sample, **add_train_data)

        output_params = Strategy.output_dictionary

        for key in test_params.keys():
            output_params[key] = test_params[key]

        return output_params

    def test_best_params(self):
        smooth_best_params = self.get_smoother_result() 
        add_test_data = {key: df.loc[self.test_sample.index] for key, df in self.additional_data.items()}
        self.get_criterion(self.test_sample, smooth_best_params, add_test_data)

        self.df_results.at[self.df_results.index[-1], 'criterion'] = self.criterion
        self.best_params_smoothed.append(smooth_best_params)

    def run_optimization(self):
        self.get_sub_samples()

        # The data is a list of pandas dataframes. So grab the starting date
        # from the first element in list and then the ending date
        # from the last element in the test_samples dataframe
        if len(self.train_samples) > 0 and len(self.test_samples) > 0:
            start = self.train_samples[0].index[0]
            start = start.strftime("%m-%d-%y %H:%M")
            end = self.test_samples[0].index[-1]
            end.strftime("%m-%d-%y %H:%M")
            print(colored(f"Dataset range --> {start} - {end}", 'green'))

        # Run the optimization
        total = len(self.train_samples)
        current = 1
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        print(colored(f"Start of optimization {formatted_now}", 'yellow'))
        
        for i, (train_sample, test_sample) in enumerate(tqdm(zip(self.train_samples, self.test_samples))):
            try:
                start = train_sample.index[0]
                start = start.strftime("%m-%d-%y %H:%M")
                end = test_sample.index[-1]
                end = end.strftime("%m-%d-%y %H:%M")
                print(colored(f"Block {current}/{total} --> {start} - {end}", 'magenta'))
            except Exception as ex:
                print("Unable to display time", ex)
            
            # Update add_train_sample and add_test_sample using the index i
            self.train_sample = train_sample
            self.test_sample = test_sample
            self.add_train_sample = self.add_train_samples[i]
            self.add_test_sample = self.add_test_samples[i]
            
            try:
                self.get_best_params_train_set()
            except Exception as e:
                print(f"Error in get_best_params_train_set: {e}")
                print(f"self.train_samples: {self.train_samples}")
                print(f"self.train_sample: {self.train_sample}")
                print(f"self.add_train_samples: {self.add_train_samples}")
                raise

            try:
                self.test_best_params()
            except Exception as e:
                print(f"Error in test_best_params: {e}") 
                print(f"self.test_samples: {self.test_samples}")
                print(f"self.test_sample: {self.test_sample}")
                print(f"self.add_test_samples: {self.add_test_samples}")
                raise

            current += 1
        
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        print(colored(f"End of optimization {formatted_now}", 'yellow'))

    def display(self):
        # Empty dataframe that will be filled by the result on each period
        df_test_result = pd.DataFrame()

        for params, test in zip(self.best_params_smoothed, self.test_samples):
            # !! Here, we can call directly the model without run again the model because the optimal weights are
            # computed already and stored into the output dictionary and so in the self.best_params_smoothed list
            self.BT = Backtest(data=test, TradingStrategy=self.TradingStrategy, parameters=params, **self.add_test_sample)
            self.BT.run()
            df_test_result = pd.concat((df_test_result, self.BT.data), axis=0)

        # Print the backtest for the period following the walk-forward method
        self.BT = Backtest(data=df_test_result, TradingStrategy=self.TradingStrategy, parameters=params, **self.add_test_sample)
        self.BT.run()
        self.BT.plot()