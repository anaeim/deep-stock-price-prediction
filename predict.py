import pandas as pd
from pathlib import Path
import math
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import load_model

from fbprophet import Prophet
from neuralprophet import NeuralProphet
from prophet.serialize import model_to_json, model_from_json

import dataloader
import models
from config import parse_args


def main():
    """Main function for running machine learning models and calculating Mean Squared Error (MSE)

    This script loads data, selects a machine learning model based on the specified command-line arguments,
    and calculates the MSE between the predicted and actual values.
    """

    args = parse_args()
    df_path = Path().cwd() / "data" / f"{args.dataset}.csv" 
    df = pd.read_csv(df_path)

    if args.ml_model == "lstm":
        data_loader = dataloader.lstm.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = load_model('./model_keeper/lstm_model.h5')
        y_predict = model.predict(X_test)

    if args.ml_model == 'prophet':
        data_loader = dataloader.neural_prophet.DataLoader(df, args)
        df, df_train, df_val = data_loader.load_data()
        _save_model_dir_path = Path().cwd() / "model_keeper" / "prophet_model.json"
        with _save_model_dir_path.open('r') as fh:
            model = model_from_json(fh.read())
        y_test,y_predict = model.predict(df)

    if args.ml_model == 'neuralprophet':
        data_loader = dataloader.neural_prophet.DataLoader(df, args)
        df, df_train, df_val = data_loader.load_data()
        _save_model_dir_path = Path().cwd() / "model_keeper" / "neural_prophet_model.json"
        with _save_model_dir_path.open('r') as fh:
            model = model_from_json(fh.read())
        y_test,y_predict = model.predict(df)

    print(f"MSE: {math.sqrt(mean_squared_error(y_test,y_predict))}")


if __name__ == "__main__":
    main()