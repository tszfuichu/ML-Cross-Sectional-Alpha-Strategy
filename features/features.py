import numpy as np

class FeatureEngine:

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

    @staticmethod
    def market_return(close_index, period=5):
        mar_return = close_index.pct_change(periods=period)
        mar_return = mar_return.dropna(how = "all")
        return mar_return

    @staticmethod
    def vix_zscore(vix, period=20):
        mean = vix.rolling(period).mean()
        std = vix.rolling(period).std()
        z_score = (vix - mean) / std
        z_score = z_score.dropna(how="all")
        return z_score

    @staticmethod
    def cross_sectional_zscore(df):
        df = df.replace([np.inf, -np.inf], np.nan)

        mean = df.mean(axis=1)
        std = df.std(axis=1)

        std = std.replace(0, np.nan)

        z = df.sub(mean, axis=0).div(std, axis=0)

        return z