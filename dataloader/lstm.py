import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """A class for loading and preprocessing time series data for machine learning.

    ...

    Attributes
    ----------
        ts : pandas.DataFrame
            The time series data to be loaded and preprocessed
        args : Namespace
            A namespace containing various configuration parameters
        train_data : pandas.DataFrame
            The training data after splitting
        test_data : pandas.DataFrame
            The testing data after splitting
        X_train : numpy.ndarray
            The feature matrix for training
        X_test : numpy.ndarray
            The feature matrix for testing
        y_train : numpy.ndarray
            The target values for training
        y_test : numpy.ndarray
            The target values for testing

    Methods
    -------
        scale_transform():
            Scales and transforms the time series data to a specified range (0 to 1)

        split_data():
            Splits the time series data into training and testing sets

        create_dataset(dataset, time_step=1):
            Creates input features and target values for time series prediction

        create_train_test_dataset(self):
            Creates feature matrices and target vectors for training and testing

        load_data(self):
            Loads, preprocesses, and splits the data, returning training and testing sets

    """

    def __init__(self, ts, args):
        """
        Parameters
        ----------
        ts : pandas.DataFrame
            The time series data to be loaded and preprocessed
        args : Namespace
            A namespace containing various configuration parameters
        """

        self.ts = ts
        self.args = args
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def scale_transform(self):
        """Scales and transforms the time series data to a specified range (0 to 1)
        """

        self.ts = self.ts['Adj Close']
        scaler=MinMaxScaler(feature_range=(0,1))
        self.ts = scaler.fit_transform(np.array(self.ts).reshape(-1,1))

    def split_data(self):
        """Splits the time series data into training and testing sets
        """

        training_size = int(len(self.ts)*self.args.test_size)
        test_size = len(self.ts)-training_size
        self.train_data, self.test_data = self.ts[0:training_size,:],self.ts[training_size:len(self.ts),:1]

    @staticmethod
    def create_dataset(dataset, time_step=1):
        """Creates input features and target values for time series prediction

        Parameters
        ----------
        dataset : pandas.DataFrame
            The time series data
        time_step : int, optional
            The time step to create datasets from time series data (default is 1)

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Feature array, target array
        """

        dataX, dataY = [], []

        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  #0-100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])

        return np.array(dataX), np.array(dataY)

    def create_train_test_dataset(self):
        """Creates feature matrices and target vectors for training and testing
        """

        self.X_train, self.y_train = self.create_dataset(self.train_data, self.args.time_step)
        self.X_test, self.y_test = self.create_dataset(self.test_data, self.args.time_step)

    def load_data(self):
        """Loads, preprocesses, and splits the data, returning training and testing sets

        Returns
        -------
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
            feature array for training, feature array for testing, target array for training, target array for testing
        """

        self.scale_transform()
        self.split_data()
        self.create_train_test_dataset()

        return self.X_train, self.X_test, self.y_train, self.y_test