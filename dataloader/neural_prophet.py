import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from neuralprophet import NeuralProphet


class DataLoader:
    """A class for loading and preprocessing time series data for neuralprophet and prophet models

    ...

    Attributes
    ----------
    args : Namespace
        A namespace object containing configuration settings for data preprocessing
    df : pandas.DataFrame
        The time series data to be loaded and preprocessed
    df_train : pandas.DataFrame
        The time series data for training
    df_val : pandas.DataFrame
        The time series data for validation

    Methods
    -------
    rename():
        Renames the columns of the DataFrame to match the expected column names for NeuralProphet
    split_data():
        Splits the input DataFrame into training and validation subsets using NeuralProphet's split_df method
    load_data():
        Preprocesses the data by renaming columns and splitting it into training and validation sets
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

        self.args = args
        self.df = df
        self.df_train = None
        self.df_test = None

    def rename(self):
        """Renames the columns of the DataFrame to match the expected column names for NeuralProphet
        """

        self.df = self.df.rename(columns = {"Date":"ds","Adj Close":"y"})

    def split_data(self):
        """Splits the input DataFrame into training and validation subsets using NeuralProphet's split_df method
        """

        model = NeuralProphet()
        self.df_train, self.df_val = model.split_df(self.df, valid_p=self.args.test_size, freq='D')

    def load_data(self):
        """Preprocesses the data by renaming columns and splitting it into training and validation sets

        Returns
        -------
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
            The time series data, the time series data for training, the time series data for validation
        """

        self.rename()
        self.split_data()
        return self.df, self.df_train, self.df_val
