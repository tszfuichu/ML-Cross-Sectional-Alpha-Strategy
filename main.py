from data.get_ticker_data import (
    get_sp500_universe,
    download_market_data,
    clean_data
)
from features.features import FeatureEngine
from features.before_modelling import Before_Modelling
# from features.Modelling import Modelling
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

start = "2018-01-01"
end = "2025-01-01"
split_date = "2023-01-01"

# get the universe tickers
tickers = get_sp500_universe(start, end)

# download the data of all tickers
df = download_market_data(tickers, start, end, filename="data/sp500_2018_2024.pkl")
spy = download_market_data("SPY", start, end, filename="data/spy_2018_2024.pkl")
vix = download_market_data("^VIX", start, end, filename="data/vix_2018_2024.pkl")


# Clean data
close, volume, high, low = clean_data(df)
spy_close = spy["Close"].squeeze()
vix_close = vix["Close"].squeeze()

# Features Engineering

stocks_raw_features = FeatureEngine.get_stocks_raw_feature(close, volume, high, low)


# shifting 1 aviod look-ahead bias
stocks_shifted_features = FeatureEngine.shift_features(stocks_raw_features, periods=1)

# Cross‑Sectional Standardization
cs_features = FeatureEngine.get_cs_z_score(stocks_shifted_features, min_stocks=50, exclude="vol_z")

# Create Forward Return(target)
forward_return_20d = close.pct_change(periods=20, fill_method=None).shift(-20)

# Convert to a Panel
panel_df = Before_Modelling.panel_construction(cs_features, forward_return_20d)

# Train / Test Split for panel data and cs_features
train_panel_df,test_panel_df,cs_features_train = Before_Modelling.split_panel_data(panel_df,cs_features,split_date=split_date)

# Calculate the IC and T-stat for each feature in training data
ic_result = Before_Modelling.compute_ic_tstat(cs_features_train, train_panel_df)
print(ic_result)

# Filter the feature by |T-stat| < 1.3 and correlation > 0.7,  We get ['mom5', 'mom20', 'vol_z'] for the feature that modelling
filter_ic_features = Before_Modelling.feature_pre_selection(ic_result, cs_features_train, threshold= 1.3, corr_threshold= 0.7)
print(filter_ic_features)

# Incremental_ic_test
incremental_ic_test = Before_Modelling.incremental_ic_test(cs_features_train,train_panel_df,filter_ic_features)
print(incremental_ic_test)

# Based on incremental_ic_test, mom5 + mom20 have the most prediction power and T-stat, so we select the mom5 and mom20 as the features
selected_features = ['mom5', 'mom20']
selected_cs_features = {f: cs_features[f] for f in selected_features}

#　After decay_result, by using ["mom5","mom20"], the peak of holding days is 27 Days, after 27th day, it start to decay
decay_result = Before_Modelling.ic_decay(close,selected_cs_features,selected_features,period=35)

# Create Optimal Forward Return and the new panel 27 days (target)
forward_return_27d = close.pct_change(periods=27, fill_method=None).shift(-27)
panel_27 = Before_Modelling.panel_construction(selected_cs_features, forward_return_27d)
train_27, test_27, _ = Before_Modelling.split_panel_data(panel_27, selected_cs_features, split_date=split_date, verbose=False)

# Create signal
test_27 = test_27.copy()
test_27["signal"] = (test_27[selected_features].mean(axis=1))

# Regime test
regime_result = Before_Modelling.regime_conditon_test(test_27,spy_close,vix_close)



