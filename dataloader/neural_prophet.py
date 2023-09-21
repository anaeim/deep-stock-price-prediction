import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from neuralprophet import NeuralProphet


class DataLoader:

    def __init__(self, df, args):
        self.args = args
        self.df = df
        self.df_train = None
        self.df_test = None

    def rename(self):
        self.df = self.df.rename(columns = {"Date":"ds","Adj Close":"y"})

    def split_data(self):


        model = NeuralProphet()
        self.df_train, self.df_val = model.split_df(self.df, valid_p=self.args.test_size, freq='D')

    def load_data(self):
        self.rename()
        self.split_data()
        return self.df, self.df_train, self.df_val
