import pandas as pd
from pathlib import Path
import pickle
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
        y_pred = model.predict(X_test)

    if args.ml_model == 'prophet':
        data_loader = dataloader.neural_prophet.DataLoader(df, args)
        df, df_train, df_val = data_loader.load_data()
        _save_model_dir_path = Path().cwd() / "model_keeper" / "prophet_model.json"
        with _save_model_dir_path.open('r') as fh:
            model = model_from_json(fh.read())
        y_test,y_pred = model.predict(df)

    if args.ml_model == 'neuralprophet':
        data_loader = dataloader.neural_prophet.DataLoader(df, args)
        df, df_train, df_val = data_loader.load_data()
        _save_model_dir_path = Path().cwd() / "model_keeper" / "neural_prophet_model.json"
        with _save_model_dir_path.open('r') as fh:
            model = model_from_json(fh.read())
        y_test,y_pred = model.predict(df)

    if args.ml == 'xgboost':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        _save_model_dir_path = models.xgboost.save_model_dir_path
        with _save_model_dir_path.open('rb') as fh:
            model = pickle.load(fh)
        y_pred = model.predict(X_test).to_numpy().reshape(-1,)
        y_test = y_test.to_numpy().reshape(-1,)

    if args.ml == 'lstm_multivariate':
        data_loader = data_loader.lstm_multivariate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        _save_model_dir_path = models.lstm_multivariate.save_model_dir_path
        model = load_model(_save_model_dir_path)
        y_pred = model.predict(X_test).reshape(-1,)
        y_test = y_test.reshape(-1,)

    if args.ml == 'lightgbm':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        _save_model_dir_path = models.lightgbm.save_model_dir_path
        with _save_model_dir_path.open('rb') as fh:
            model = pickle.load(fh)
        y_pred = model.predict(X_test).to_numpy().reshape(-1,)
        y_test = y_test.to_numpy().reshape(-1,)

    if args.ml == 'ensemble_XGBoost_lightgbm':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        _save_model_dir_path = models.ensemble.save_model_dir_path
        with _save_model_dir_path.open('rb') as fh:
            model = pickle.load(fh)
        y_pred = model.predict(X_test)
        y_test = y_test.to_numpy().reshape(-1,)

    if args.ml == 'ensemble_XGBoost_lightgbm_lstm_multivariate':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        _save_model_dir_path = models.ensemble.save_model_dir_path
        with _save_model_dir_path.open('rb') as fh:
            model = pickle.load(fh)
        y_pred = model.predict(X_test)
        y_test = y_test.to_numpy().reshape(-1,)

    if args.ml == 'ensemble_lightgbm_lstm_multivariate':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        _save_model_dir_path = models.ensemble.save_model_dir_path
        with _save_model_dir_path.open('rb') as fh:
            model = pickle.load(fh)
        y_pred = model.predict(X_test)
        y_test = y_test.to_numpy().reshape(-1,)


    print(f"MSE: {math.sqrt(mean_squared_error(y_test,y_pred))}")


if __name__ == "__main__":
    main()