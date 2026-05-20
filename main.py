from data.get_ticker_data import (
    get_sp500_universe,
    download_market_data,
    clean_data
)
from features.features import FeatureEngine
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

mom5 = FeatureEngine.momentum(close, period= 5)
mom20 = FeatureEngine.momentum(close, period= 20)
mom60 = FeatureEngine.momentum(close, period= 60)
ma20 = FeatureEngine.moving_average(close, period = 20)
rsi14 = FeatureEngine.rsi(close,period= 14)
rolling_vol_20 = FeatureEngine.rolling_volatility(close, period=20)
rolling20_hl_range = FeatureEngine.rolling_high_low_range(high, low, close, period=20)
rolling20_volume_z_score = FeatureEngine.rolling_volume_zscore(volume,period=20)
volume_change = FeatureEngine.volume_change(volume, period=5)

# Market Contest
market_return_5d = FeatureEngine.market_return(spy_close, period= 5)
vix_zscore_20d = FeatureEngine.vix_zscore(vix_close, period=20)

# shifting 1 aviod look-ahead bias
mom5 = mom5.shift(1)
mom20 = mom20.shift(1)
mom60 = mom60.shift(1)
ma20 = ma20.shift(1)
rsi14 = rsi14.shift(1)
rolling_vol_20 = rolling_vol_20.shift(1)
rolling20_hl_range = rolling20_hl_range.shift(1)
rolling20_volume_z_score = rolling20_volume_z_score.shift(1)
volume_change = volume_change.shift(1)

market_return_5d = market_return_5d.shift(1)
vix_zscore_20d = vix_zscore_20d.shift(1)


# Create Forward Return(target)
forward_return_5d = close.pct_change(periods=5).shift(-5)

# Cross‑Sectional Standardization
mom5_cs = FeatureEngine.cross_sectional_zscore(mom5)
mom20_cs = FeatureEngine.cross_sectional_zscore(mom20)
mom60_cs = FeatureEngine.cross_sectional_zscore(mom60)
ma20_cs = FeatureEngine.cross_sectional_zscore(ma20)
rsi14_cs = FeatureEngine.cross_sectional_zscore(rsi14)
rolling_vol_cs = FeatureEngine.cross_sectional_zscore(rolling_vol_20)
rolling_hl_range_cs = FeatureEngine.cross_sectional_zscore(rolling20_hl_range)
rolling20_volume_cs = rolling20_volume_z_score
volume_change_cs = FeatureEngine.cross_sectional_zscore(volume_change)


# Broadcast Market & VIX to Panel Shape
market_panel = pd.concat(
    [market_return_5d] * close.shape[1],
    axis=1
)
market_panel.columns = close.columns

vix_panel = pd.concat(
    [vix_zscore_20d] * close.shape[1],
    axis=1
)
vix_panel.columns = close.columns

print(market_panel, vix_panel)