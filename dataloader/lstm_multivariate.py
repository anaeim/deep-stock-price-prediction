from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

window = 24

class DataLoader:
    """A class for loading and preprocessing time series data for machine learning.

    ...

    Attributes
    ----------
        ts : pandas.DataFrame
            The time series data to be loaded and preprocessed
        args : Namespace
            A namespace containing various configuration parameters
        df_tain : pandas.DataFrame
            The training data after splitting
        df_test : pandas.DataFrame
            The testing data after splitting
        X_train : pandas.DataFrame
            The feature matrix for training
        X_test : pandas.DataFrame
            The feature matrix for testing
        y_train : pandas.DataFrame
            The target values for training
        y_test : pandas.DataFrame
            The target values for testing

    Methods
    -------
        split_data(split_date='2018-01-01'):
            Splits the input DataFrame into training and testing sets based on a specified date
        create_input_features(df, target='Adj Close')
            Creates input features and target variable from the input DataFrame
        apply_input_features()
            Creates input features and target variables for both training and testing sets
        create_window_data(X, Y)
            Creates windowed data for time-series analysis
        apply_window_data()
            Applies the windowing technique to training and testing data
        load_data()
            Loads, preprocesses, and splits the data, returning training and testing sets
    """

    def __init__(self, df, args):
        """
        Parameters
        ----------
        ts : pandas.DataFrame
            The time series data to be loaded and preprocessed
        args : Namespace
            A namespace containing various configuration parameters
        """

        self.df = df
        self.args = args
        self.window = window
        self.df_tain = None
        self.df_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, split_date='2018-01-01'):
        """Splits the input DataFrame into training and testing sets based on a specified date

        Parameters
        ----------
        split_date : str, optional
            The date used for splitting the data into training and testing sets. The default is '2018-01-01'
        """

        self.df_train = self.df.loc[self.df.index <= split_date]
        self.df_test = self.df.loc[self.df.index > split_date]
        print(f"{len(self.df_train)} days of training data \n {len(self.df_test)} days of testing data ")

    @staticmethod
    def create_input_features(df, target='Adj Close'):
        """Creates input features and target variable from the input DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing time-series data
        target : str, optional
            The name of the target variable to be predicted. The default is 'Adj Close'

        Returns
        -------
        X : pandas.DataFrame
            The feature matrix
        y : pandas.Series
            The target variable
        """

        df.set_index('date', inplace=True)
        X = df.drop(['date'], axis=1)
        if target:
            y = df[target]
            X = X.drop([target], axis=1)
            return X, y

    def apply_input_features(self):
        self.X_train, self.y_train = self.create_input_features(self.df_tain)
        self.X_test, self.y_test = self.create_input_features(self.df_test)

    def create_window_data(self, X, Y):
        """Creates windowed data for time-series analysis

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix
        Y : pandas.Series
            The target variable

        Returns
        -------
        X_windowed : numpy.ndarray
            The windowed feature matrix
        Y_windowed : numpy.ndarray
            The corresponding target variable for the windowed data
        """

        x = []
        y = []
        for i in range(self.window-1, len(X)):
            x.append(X[i-self.window+1:i+1])
            y.append(Y[i])
        return np.array(x), np.array(y)
    
    def apply_window_data(self):
        """Applies the windowing technique to training and testing data
        """

        X_w = np.concatenate((self.X_train, self.X_test))
        y_w = np.concatenate((self.y_train, self.y_test))

        X_w, y_w = self.window_data(X_w, y_w)
        self.X_train_w = X_w[:-len(self.X_test)]
        self.y_train_w = y_w[:-len(self.X_test)]
        self.X_test_w = X_w[-len(self.X_test):]
        self.y_test_w = y_w[-len(self.X_test):]

    def load_data(self):
        """Loads, preprocesses, and splits the data, returning training and testing sets

        Returns
        -------
        X_train_w : numpy.ndarray
            Feature array for training
        X_test_w : numpy.ndarray
            Feature array for testing
        y_train_w : numpy.ndarray
            Target array for training
        y_test_w : numpy.ndarray
            Target array for testing
        """

        self.split_data()
        self.apply_time_features()
        self.scale_transform()

        return self.X_train_w, self.X_test_w, self.y_train_w, self.y_test_w