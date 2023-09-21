from neuralprophet import NeuralProphet
from prophet.serialize import model_to_json, model_from_json

from pathlib import Path

class Model:
    """A class representing a NeuralProphet model for time series forecasting

    ...

    Attributes
    ----------
    args : Namespace
        A namespace containing model configuration arguments
    save_model_dir_path : Path
       The path to save the serialized model
    model : NeuralProphet
        NeuralProphet Model for time series forecasting

    Methods
    -------
    forward()
        Initializes the NeuralProphet model
    fit(df_train, df_val)
        Fits the NeuralProphet model to the training data
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
        self.save_model_dir_path = Path().cwd() / "model_keeper" / "neural_prophet_model.json"

    def forward(self):
        """Initializes the NeuralProphet model
        """

        self.model = NeuralProphet()

    def fit(self, df_train, df_val):
        """Fits the NeuralProphet model to the training data

        Parameters
        ----------
        df_train : pandas.DataFrame
            Input training data (features)
        df_val : pandas.DataFrame
            Input validation data (features)
        """

        self.model.fit(df_train, validation=df_val, freq='D', epochs=self.args.epochs)

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

        df_future = self.model.make_future_dataframe(df, periods=self.args.time_step, n_historic_predictions=len(df))
        y_predict = self.model.predict(df_future)
        
        return  df_future.to_numpy()[['y']].reshape(-1,), y_predict[['yhat']].to_numpy().reshape(-1,)