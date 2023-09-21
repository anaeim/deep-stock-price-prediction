import pandas as pd
from pathlib import Path

import dataloader
import models
from config import parse_args


def main():
    """Main function to load data, train machine learning models, and optionally save the trained model

    This script reads a CSV dataset, performs data loading and model training based on the user's choice of machine
    learning model, and can save the trained model if specified.
    """

    args = parse_args()
    df_path = Path().cwd() / "data" / f"{args.dataset}.csv" 
    df = pd.read_csv(df_path)

    if args.ml_model_model == "lstm":
        data_loader = dataloader.lstm.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = model.lstm.Model()
        model.fit(X_train, y_train)

    if args.ml_model == 'prophet':
        data_loader = dataloader.neural_prophet.DataLoader(df, args)
        df, df_train, df_val = data_loader.load_data()
        model = models.prophet.Model(args)
        model.fit(df_train, df_val)

    if args.ml_model == 'neuralprophet':
        data_loader = dataloader.neural_prophet.DataLoader(df, args)
        df, df_train, df_val = data_loader.load_data()
        model = models.neural_prophet.Model(args)
        model.fit(df_train)

    if args.ml_model == 'xgboost':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = models.xgboost.Model(args)
        model.fit(X_train, y_train)

    if args.ml_model == 'lstm_multivariate':
        data_loader = data_loader.lstm_multivariate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = models.lstm_multivariate.Model(args)
        model.fit(X_train, y_train)

    if args.ml_model == 'lightgbm':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = models.lightgbm.Model(args)
        model.fit(X_train, y_train)

    if args.ml_model == 'ensemble_XGBoost_lightgbm':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = models.ensemble.Model(args)
        model.fit(X_train, y_train)

    if args.ml_model == 'ensemble_XGBoost_lightgbm_lstm_multivariate':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = models.ensemble.Model(args)
        model.fit(X_train, y_train)

    if args.ml_model == 'ensemble_lightgbm_lstm_multivariate':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = models.ensemble.Model(args)
        model.fit(X_train, y_train)

    if args.ml_model == 'ensemble_XGBoost_lstm_multivariate':
        data_loader = data_loader.multi_variate.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = models.ensemble.Model(args)
        model.fit(X_train, y_train)

    if args.enable_save_model:
        model.save_model()

if __name__ == "__main__":
    main()