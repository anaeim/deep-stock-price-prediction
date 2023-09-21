from pathlib import Path
import pickle

from models import xgboost, lstm_multivariate, lightgbm

save_model_dir_path = Path().cwd() / "model_keeper" / "ensemble_model.pkl"

class Model:
    """A class representing a XGBoost Regressor model for time series forecasting

    ...

    Attributes
    ----------
    args : Namespace
        A namespace containing model configuration arguments
    save_model_dir_path : Path
       The path to save the serialized model
    model_xgb : xgb.XGBRegressor
        xgb.XGBRegressor Model for time series forecasting
    model_lstm_multi_var : tensoeflow.keras.Model
        tensoeflow.keras.Model Model for time series forecasting
    model_lightgbm : lightgbm.LGBMRegressor
        lightgbm.LGBMRegressor Model for time series forecasting

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
        self.prediction_dict = dict()

        self.model_xgb = None
        self.model_lstm_multi_var = None
        self.model_lightgbm = None


    def forward(self):
        """Initializes the model
        """
        _model_xgb = xgboost.Model(self.args)
        self.model_xgb = _model_xgb.forward()

        _model_lstm_multi_var = lstm_multivariate.Model(self.args)
        self.model_lstm_multi_var = _model_lstm_multi_var.forward()

        _model_lightgbm = lightgbm.Model(self.args)
        self.model_lightgbm = _model_lightgbm.forward()


    def fit(self, X_train, y_train):
        """Fits the model to the training data

        Parameters
        ----------
        X_train : pandas.DataFrame
            Input training data (features)
        y_train : pandas.DataFrame
            Target for the training data
        """

        self.model_xgb.fit(X_train, y_train, verbose=False)
        self.model_lstm_multi_var.fit(X_train, y_train)
        self.model_lightgbm.fit(X_train, y_train)

    def save_model(self):
        """Saves the trained model
        """

        with self.save_model_dir_path.open('wb') as fh:
            pickle.dump(self.model, fh)

    def predict(self, X_test):
        """Makes predictions using the trained model

        Parameters
        ----------
        X_test : pandas.DataFrame
            Input testing data (features)

        Returns
        -------
        numpy.ndarray
            Predicted target for the testing data
        """

        self.prediction_dict['XGBoost'] = self.model_xgb.predict(X_test).to_numpy().reshape(-1,)
        self.prediction_dict['lstm_multivariate'] = self.model_lstm_multi_var.predict(X_test).reshape(-1,)
        self.prediction_dict['lightgbm'] = self.model_lightgbm.predict(X_test).to_numpy().reshape(-1,)

        if self.args == 'ensemble_XGBoost_lightgbm':
            return (self.prediction_dict['XGBoost'] + self.prediction_dict['lightgbm']) / 2

        if self.args == 'ensemble_XGBoost_lightgbm_lstm_multivariate':
            return (self.prediction_dict['XGBoost'] + self.prediction_dict['lightgbm'] + self.prediction_dict['lstm_multivariate']) / 3

        if self.args == 'ensemble_lightgbm_lstm_multivariate':
            return (self.prediction_dict['lightgbm'] + self.prediction_dict['lstm_multivariate']) / 2

        if self.args == 'ensemble_XGBoost_lstm_multivariate':
            return (self.prediction_dict['XGBoost'] + self.prediction_dict['lstm_multivariate']) / 2