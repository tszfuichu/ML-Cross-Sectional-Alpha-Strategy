import numpy as np

class FeatureEngine:

    # Staticmethod(Moving Average, Momentum, RSI, Rolling Volatility, Rolling High Low Range, Rolling Volume Z score, Volume Change)

    @staticmethod
    def moving_average(df, period=20):
        return df.rolling(period).mean().dropna(how="all")

    @staticmethod
    def momentum(df, period=20):
        return df.pct_change(periods=period, fill_method=None).dropna(how="all")

    @staticmethod
    def rsi(df, period = 14):
        # Get the price difference
        diff = df.diff()

        # Separate gain and loss
        gain = diff.clip(lower = 0)
        loss = -diff.clip(upper = 0)

        # Step 3: rolling mean per stock (column-wise)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        # 4. Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.dropna(how="all")

        return rsi

    @staticmethod
    def rolling_volatility(df, period, annualize = True):
        log_ret = np.log(df).diff()
        vol = log_ret.rolling(period, min_periods=period).std()
        if annualize:
            vol = vol * np.sqrt(252)
        return vol.dropna(how="all")

    @staticmethod
    def rolling_high_low_range(high,low,close,period = 20):
        daily_range = (high - low) / close
        hl_range = daily_range.rolling(period, min_periods=period).mean()
        hl_range = hl_range.dropna(how="all")
        return hl_range

    @staticmethod
    def rolling_volume_zscore(volume, period=20):
        rolling_mean = volume.rolling(window=period, min_periods=period).mean()
        rolling_std = volume.rolling(window=period, min_periods=period).std()

        zscore = (volume - rolling_mean) / rolling_std
        zscore = zscore.dropna(how="all")
        return zscore

    @staticmethod
    def volume_change(volume, period=5):
        volume_change = volume.pct_change(periods=period)
        volume_change = volume_change.dropna(how="all")
        return volume_change

    #======================================================
    # Get all the Feature by using the static function above

    @staticmethod
    def get_stocks_raw_feature(close, volume, high, low):
        return {
            "mom5": FeatureEngine.momentum(close, 5),
            "mom20": FeatureEngine.momentum(close, 20),
            "mom60": FeatureEngine.momentum(close, 60),
            "ma20": FeatureEngine.moving_average(close, 20),
            "rsi14": FeatureEngine.rsi(close, 14),
            "vol": FeatureEngine.rolling_volatility(close, 20),
            "hl_range": FeatureEngine.rolling_high_low_range(high, low, close, 20),
            "vol_z": FeatureEngine.rolling_volume_zscore(volume, 20),
            "vol_change": FeatureEngine.volume_change(volume, 5),
        }


    @staticmethod
    def shift_features(features: dict, periods: int = 1):
        shifted = {}

        for name, df in features.items():
            shifted[name] = df.shift(periods)
            shifted[name] = shifted[name].dropna(how= "all")

        return shifted

    # ===========================================
    # Cross_Sectional_Z_score Calcualtion

    @staticmethod
    def cross_sectional_zscore(df, min_stocks=50):

        # Count How many vaild stocks
        count = df.count(axis=1)

        # Replace infinities first
        df = df.replace([np.inf, -np.inf], np.nan)

        # Cross-sectional mean & std (per date)
        mean = df.mean(axis=1, skipna=True)
        std = df.std(axis=1, skipna=True)

        # Avoid division by zero
        std = std.replace(0, np.nan)

        # Broadcasting subtraction/division
        z = df.sub(mean, axis=0).div(std, axis=0)
        z.loc[count < min_stocks, :] = np.nan

        return z

    # ====================================
    # Apply the Cross_Sectional_Z_score Function to Stock list

    @staticmethod
    def get_cs_z_score(feature: dict, min_stocks=50, exclude=None):

        exclude = exclude or []
        cs_dict = {}

        for name,df in feature.items():
            if name in exclude:
                cs_dict[name] = df
            else:
                cs_dict[name] = FeatureEngine.cross_sectional_zscore(df, min_stocks=min_stocks)

        return cs_dict
