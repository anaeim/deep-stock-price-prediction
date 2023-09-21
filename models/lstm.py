### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,BatchNormalization
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.callbacks import EarlyStopping

import math
from sklearn.metrics import mean_squared_error


class Model:
    """Stacked LSTM Model for Time Series Forecasting

    ...

    Attributes
    ----------
    args : Namespace
        A namespace containing configuration options
    model : tensorflow.keras.Model
        deep learning based on tensorflow library

    Methods
    -------
    forward()
        Builds the model architecture
    fit(X_train, y_train)
        Trains the model on the provided training data
    save_model()
        Saves the trained model to a file
    predict(X_test)
        Makes predictions on the provided test data
    __call__(X_train, y_train)
        Trains the model and optionally save it when the instance is called
    """

    def __init__(self, args):
        self.args = args
        self.model = None

    def forward(self):
        """Builds the model architecture
        """

        self.model=Sequential()
        # Adding first LSTM layer
        self.model.add(LSTM(units=100, return_sequences=True, input_shape=(100,1)))
        self.model.add(Dropout(0.2))
        # second LSTM layer 
        self.model.add(LSTM(units=100, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding third LSTM layer 
        self.model.add(LSTM(units=100, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding fourth LSTM layer
        self.model.add(LSTM(units=100, return_sequences=True))
        self.model.add(Dropout(0.2))
        # Adding fifth LSTM layer a
        self.model.add(LSTM(units=100))
        self.model.add(Dropout(0.2))
        # Adding the Output Layer
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print(self.model.summary())

    def fit(self, X_train, y_train):
        """Trains the model on the provided training data

        Parameters
        ----------
        X_train : numpy.ndarray
            Input training data (features)
        y_train : numpy.ndarray
            Target training data (labels)
        """

        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, 
                verbose=1, mode='auto', restore_best_weights=True)

        history = self.model.fit(X_train,y_train,
                callbacks=[monitor],verbose=1,epochs=self.args.epochs, batch_size=self.args.batch_size)

    def save_model(self):
        """Saves the trained model to a file
        """

        self.model.save('./model_keeper/lstm_model.h5')

    def predict(self, X_test):
        """Makes predictions on the provided test data

        Parameters
        ----------
        X_train : numpy.ndarray
            Input testing data (features)

        Returns
        -------
        numpy.ndarray
            Target testing data (labels)
        """

        y_predict = self.model.predict(X_test)
        return y_predict

    def __call__(self, X_train, y_train):
        """Trains the model and optionally save it when the instance is called
        """

        self.fit(X_train, y_train)

        if self.args.enable_save_model:
            self.save_model()