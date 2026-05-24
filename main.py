from data.get_ticker_data import (
    get_sp500_universe,
    download_market_data,
    clean_data
)
from features.features import FeatureEngine
from features.before_modelling import Before_Modelling
# from features.Modelling import Modelling
import numpy as np
import pandas as pd

start = "2018-01-01"
end = "2025-01-01"

# get the universe tickers
tickers = get_sp500_universe(start, end)

# download the data of all tickers
df = download_market_data(tickers, start, end, filename="data/sp500_2018_2024.pkl")
spy = download_market_data("SPY", start, end, filename="data/spy_2018_2024.pkl")
vix = download_market_data("^VIX", start, end, filename="data/vix_2018_2024.pkl")


# Clean data
close, volume, high, low = clean_data(df)
spy_close = spy["Close"]
vix_close = vix["Close"]

# Features Engineering

stocks_raw_features = FeatureEngine.get_stocks_raw_feature(close, volume, high, low)

# Market Context
market_features = FeatureEngine.get_market_raw_feature(spy_close, vix_close)


# shifting 1 aviod look-ahead bias
stocks_shifted_features = FeatureEngine.shift_features(stocks_raw_features, periods=1)
market_shifted_features = FeatureEngine.shift_features(market_features, periods=1)

# Create Forward Return(target)
forward_return_5d = close.pct_change(periods=5).shift(-5)

# Cross‑Sectional Standardization
cs_features = FeatureEngine.get_cs_z_score(stocks_shifted_features, min_stocks=50, exclude="vol_z")

# Convert to a Panel
panel_df = Before_Modelling.panel_construction(cs_features, forward_return_5d)
print(panel_df)

# Train / Test Split for panel data and cs_features
train_panel_df,test_panel_df,cs_features_train = Before_Modelling.split_panel_data(panel_df,cs_features,split_date="2024-01-01")
print(train_panel_df, test_panel_df, cs_features_train)

# Calculate the IC and T-stat for each feature in training data
ic_result = Before_Modelling.compute_ic_tstat(cs_features_train, train_panel_df)
print(ic_result)

# Filter the feature by |T-stat| < 1.3 and correlation > 0.7,  We get ['mom5', 'mom20', 'vol_z'] for the feature that modelling
filter_ic_features = Before_Modelling.feature_pre_selection(ic_result, cs_features_train, threshold= 1.3, corr_threshold= 0.7)
print(filter_ic_features)
