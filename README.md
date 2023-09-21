# deep-time-series-forecasting-models-for-stock-price-prediction

<img
  src="images/stoch_data_graph.jpg"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 600px">

# Intro
A collection of prominent and SOTA time-series forecatsing models, including deep, uni-variate, multi-variate and ensemble models.

# Installation
To get started, you'll need Python and pip installed.

1. Clone the Git repository
```
git clone https://github.com/anaeim/deep-stock-price-prediction.git
```

2. Navigate to the project directory
```
cd deep-stock-price-prediction
```

3. Create directories for the stock data and the trained models<br>
   You can download Apple (AAPL) Historical Stock Data from [this Kaggle web page](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data) and Tesla (TSLA) stock data from [this kaggle web page](https://www.kaggle.com/code/debashis74017/time-series-forecasting-tesla-stock/notebook).
```
mkdir data
mkdir model_keeper
```

1. Install the requirements
```
pip install -r requirements.txt
```

# Models
- uni-variate LSTM
- [Prophet](https://github.com/facebook/prophet)
- [NeuralProphet](https://github.com/ourownstory/neural_prophet)
- multi-variate LSTM
- [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
- [LightGBM](https://github.com/microsoft/LightGBM)
- ensemble models:
  - XGBoost + multi-variate LSTM
  - XGBoost + LightGBM
  - LightGBM + multi-variate LSTM
  - XGBoost + LightGBM + multi-variate LSTM



# Training
```
python train.py --dataset AAPL \
    --ml-model ensemble_XGBoost_lstm_multivariate \
    --test-size 0.35 \
    --time-stamp 100 \
    --epoch 1000 \
    --batch-size 84 \
    --enable-save-model
```


The meaning of the flags:
* ``--dataset-path``: the directory that contains the dataset
* ``--ml-model``: the Machine Learning (ML) model that we use for time-series forecasting
* ``--time-stamp``: the time stamp to create windowed datasets
* ``--test-size``, ``--epochs``, and ``--batch-size`` are the proportion of the dataset to include in the test split, number of epochs and number of training examples in each iteration of the model, respectively.
* ``enable-save-model``: to save the trained model into the ``model_keeper`` directory


# Prediction
```
python train.py --dataset AAPL \
    --ml-model ensemble_XGBoost_lstm_multivariate \
    --test-size 0.35 \
    --time-stamp 100 
```