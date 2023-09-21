import pandas as pd
from pathlib import Path

import dataloader
import models
from config import parse_args


def main():

    args = parse_args()
    df_path = Path().cwd() / "data" / f"{args.dataset}.csv" 
    df = pd.read_csv(df_path)

    if args.ml_model == "lstm":
        data_loader = dataloader.lstm.DataLoader(df, args)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        model = model.lstm.Model()
        model.fit(X_train, y_train)


        model.fit(df_train)

    if args.enable_save_model:
        model.save_model()


if __name__ == "__main__":
    main()