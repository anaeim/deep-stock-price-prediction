from pathlib import Path
import pickle

import tensorflow as tf

save_model_dir_path = Path().cwd() / "model_keeper" / "lstm_multivariate.h5"
BATCH_SIZE = 64
BUFFER_SIZE = 100
WINDOW_LENGTH = 24

class Model:
    """A class representing a XGBoost Regressor model for time series forecasting

    ...

    Attributes
    ----------
    args : Namespace
        A namespace containing model configuration arguments
    save_model_dir_path : Path
       The path to save the serialized model
    model : xgb.XGBRegressor
        xgb.XGBRegressor Model for time series forecasting

    Methods
    -------
    forward()
        Initializes the model
    fit(X_train, y_train)
        Fits the model to the training data
    save_model()
        Saves the trained model
    predict(X_test)
        Makes predictions using the trained model
    """

    def __init__(self, args):
        """
        Parameters
        ----------
        args : Namespace
            A namespace containing model configuration arguments
        """

        self.args = args
        self.save_model_dir_path = save_model_dir_path

    def forward(self):
        """Initializes the model
        """

        dropout = 0.0
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                128, input_shape=self.X_train_w.shape[-2:], dropout=dropout),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='rmsprop', loss='mae')


    def fit(self, X_train, y_train):
        """Fits the model to the training data

        Parameters
        ----------
        X_train : numpy.ndarray
            Input training data (features)
        y_train : numpy.ndarray
            Target for the training data
                """
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        model_history = self.model.fit(train_data, epochs=self.args.epochs)
        

    def save_model(self):
        """Saves the trained model
        """

        self.model.save(self.save_model_dir_path)

    def predict(self, X_test):
        """Makes predictions using the trained model

        Parameters
        ----------
        X_test : numpy.ndarray
            Input testing data (features)

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted target for the testing data
        """

        y_pred = self.model.predict(X_test)
        return  y_pred