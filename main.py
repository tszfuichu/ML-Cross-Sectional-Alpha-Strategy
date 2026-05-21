from data.get_ticker_data import (
    get_sp500_universe,
    download_market_data,
    clean_data
)
from features.features import FeatureEngine
from features.before_modelling import Before_Modelling
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

panel_df = Before_Modelling.panel_construction(cs_features, forward_return_5d)

ic_result = Before_Modelling.compute_ic_tstat(cs_features,panel_df)
print(ic_result)