from pathlib import Path
import pickle

import xgboost as xgb

save_model_dir_path = Path().cwd() / "model_keeper" / "xgboost.pkl"

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

        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)

    def fit(self, X_train, y_train):
        """Fits the model to the training data

        Parameters
        ----------
        X_train : pandas.DataFrame
            Input training data (features)
        y_train : pandas.DataFrame
            Target for the training data
        """

        self.model.fit(X_train, y_train, verbose=False)

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
        pandas.DataFrame
            Predicted target for the testing data
        """

        y_predict = self.model.predict(X_test)
        return  y_predict