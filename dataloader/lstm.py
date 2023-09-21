import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataLoader:


    def __init__(self, ts, args):

        self.ts = ts
        self.args = args
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def scale_transform(self):

        self.ts = self.ts['Adj Close']
        scaler=MinMaxScaler(feature_range=(0,1))
        self.ts = scaler.fit_transform(np.array(self.ts).reshape(-1,1))

    def split_data(self):

        training_size = int(len(self.ts)*self.args.test_size)
        test_size = len(self.ts)-training_size
        self.train_data, self.test_data = self.ts[0:training_size,:],self.ts[training_size:len(self.ts),:1]

    @staticmethod
    def create_dataset(dataset, time_step=1):

        dataX, dataY = [], []

        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  #0-100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])

        return np.array(dataX), np.array(dataY)

    def create_train_test_dataset(self):


        self.X_train, self.y_train = self.create_dataset(self.train_data, self.args.time_step)
        self.X_test, self.y_test = self.create_dataset(self.test_data, self.args.time_step)

    def load_data(self):

        self.scale_transform()
        self.split_data()
        self.create_train_test_dataset()

        return self.X_train, self.X_test, self.y_train, self.y_test