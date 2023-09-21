from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, df, args):
        self.df = df
        self.args = args
        self.df_tain = None
        self.df_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, split_date='2018-01-01'):
        self.df.set_index('date', inplace=True)

        self.df_train = self.df.loc[self.df.index <= split_date]
        self.df_test = self.df.loc[self.df.index > split_date]
        print(f"{len(self.df_train)} days of training data \n {len(self.df_test)} days of testing data ")

    @staticmethod
    def create_time_features(df, target=None):
        """
        Creates time series features from datetime index and target
        """

        df['date'] = df.index
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dayofmonth'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['amorpm'] = np.where(df['hour']<12,1,2)
        df['dayofweek'] = df['date'].dt.dayofweek
        df['weekend_status'] = np.where(df['day_of_week'].isin([5,6]),1,0)
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['sin_day'] = np.sin(df['dayofyear'])
        df['cos_day'] = np.cos(df['dayofyear'])
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear
        # lag
        df['lag_7'] = df[target].shift(7)
        df['lag_14'] = df[target].shift(14)
        # SMA
        df['SMA7'] = df[target].rolling(window=7).mean()
        df['SMA15'] = df[target].rolling(window=15).mean()
        df['SMA30'] = df[target].rolling(window=30).mean()

        X = df.drop(['date'], axis=1)
        if target:
            y = df[target]
            X = X.drop([target], axis=1)
            return X, y


    def apply_time_features(self):
        X_train, y_train = self.create_time_features(self.df_tain, 'Adj Close')
        X_test, y_test = self.create_time_features(self.df_test, 'Adj Close')

    def scale_transform(self):
        """Scales and transforms the time series data to a specified range (0 to 1)
        """

        scaler = StandardScaler()
        scaler.fit(self.X_train)
        _X_train = scaler.transform(self.X_train)
        _X_test = scaler.transform(self.X_test)
        self.X_train = pd.DataFrame(_X_train, columns=self.X_train.columns)
        self.X_test = pd.DataFrame(_X_test, columns=self.X_test.columns)

    def load_data(self):
        """Loads, preprocesses, and splits the data, returning training and testing sets

        Returns
        -------
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
            feature array for training, feature array for testing, target array for training, target array for testing
        """

        self.split_data()
        self.apply_time_features()
        self.scale_transform()

        return self.X_train, self.X_test, self.y_train, self.y_test
