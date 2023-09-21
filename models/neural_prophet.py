from neuralprophet import NeuralProphet
from prophet.serialize import model_to_json, model_from_json

from pathlib import Path

class Model:


    def __init__(self, args):

        self.args = args
        self.save_model_dir_path = Path().cwd() / "model_keeper" / "neural_prophet_model.json"

    def forward(self):

        self.model = NeuralProphet()

    def fit(self, df_train, df_val):


        self.model.fit(df_train, validation=df_val, freq='D', epochs=self.args.epochs)

    def save_model(self):

        with self.save_model_dir_path.open('w') as fh:
            fh.write(model_to_json(self.model))

    def predict(self, df):

        df_future = self.model.make_future_dataframe(df, periods=self.args.time_step, n_historic_predictions=len(df))
        y_predict = self.model.predict(df_future)
        
        return  df_future.to_numpy()[['y']].reshape(-1,), y_predict[['yhat']].to_numpy().reshape(-1,)