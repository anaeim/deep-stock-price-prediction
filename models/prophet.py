from fbprophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from pathlib import Path

class Model:
    """A class representing a Prophet model for time series forecasting

    ...

    Attributes
    ----------
    args : Namespace
        A namespace containing model configuration arguments
    save_model_dir_path : Path
       The path to save the serialized model
    model : Prophet
        Prophet Model for time series forecasting

    Methods
    -------
    forward()
        Initializes the Prophet model
    fit(df_train)
        Fits the Prophet model to the training data
    save_model()
        Saves the trained model to a JSON file
    predict(df)
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
        self.save_model_dir_path = Path().cwd() / "model_keeper" / "prophet_model.json"

    def forward(self):
        """Initializes the Prophet model
        """

        self.model = Prophet(
            growth='linear',
            seasonality_mode='multiplicative',
            holidays_prior_scale=20,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False
            ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=55
            ).add_seasonality(
                name='daily',
                period=1,
                fourier_order=15
            ).add_seasonality(
                name='weekly',
                period=7,
                fourier_order=25
            ).add_seasonality(
                name='yearly',
                period=365.25,
                fourier_order=20
            ).add_seasonality(
                name='quarterly',
                period=365.25/4,
                fourier_order=55
            ).add_country_holidays(country_name='China')

    def fit(self, df_train):
        """Fits the Prophet model to the training data

        Parameters
        ----------
        df_train : pandas.DataFrame
            Input training data (features)
        """

        self.model.fit(df_train)

    def save_model(self):
        """Saves the trained model to a JSON file
        """

        with self.save_model_dir_path.open('w') as fh:
            fh.write(model_to_json(self.model))

    def predict(self, df):
        """Makes predictions using the trained model

        Parameters
        ----------
        df : pandas.DataFrame
            Input testing data (features)

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Target of testing data (labels), predicted target of testing data
        """

        df_future = self.model.make_future_dataframe(periods=self.args.time_step, freq='D')
        y_predict = self.model.predict(df_future)
        
        return  df_future.to_numpy()[['y']].reshape(-1,), y_predict[['yhat']].to_numpy().reshape(-1,)