import pandas as pd
from scipy.stats import spearmanr
import numpy as np

class Before_Modelling:

    @staticmethod
    def panel_construction(cs_feature: dict, target: dict):
        stacked_features = {}
        for name, df in cs_feature.items():
            stacked_features[name] = df.stack()
        stacked_target = target.stack()

        panel_df = pd.DataFrame(stacked_features)
        panel_df["target"] = stacked_target
        panel_df.index.names = ["Date", "Ticker"]
        panel_df = panel_df.dropna()

        return panel_df

    @staticmethod
    def split_panel_data(panel_df,cs_features, split_date, verbose=True):

        split_date = pd.to_datetime(split_date)
        train_mask = panel_df.index.get_level_values("Date") < split_date
        test_mask = panel_df.index.get_level_values("Date") >= split_date

        train_panel = panel_df.loc[train_mask]
        test_panel = panel_df.loc[test_mask]

        if verbose:
            print("Train period:", train_panel.index.get_level_values("Date").min(),
                  "to", train_panel.index.get_level_values("Date").max())

            print("Test period:", test_panel.index.get_level_values("Date").min(),
                  "to", test_panel.index.get_level_values("Date").max())

        train_dates = train_panel.index.get_level_values("Date").unique()

        cs_features_train = {
            f: df.loc[df.index.intersection(train_dates)]
            for f, df in cs_features.items()
        }

        return train_panel,test_panel,cs_features_train


    @staticmethod
    def compute_ic_tstat(cs_feature, panel_df):
        print("\nComputing Feature IC...")

        feature_cols = list(cs_feature.keys())
        ic_results = {}

        for feature in feature_cols:
            daily_ic = (
                panel_df[[feature, "target"]]
                    .groupby(level="Date")
                    .apply(lambda x: spearmanr(x[feature], x["target"])[0])
            )

            mean_ic = daily_ic.mean()
            std_ic = daily_ic.std()
            t_stat = mean_ic / std_ic * np.sqrt(len(daily_ic))

            ic_results[feature] = {
                "Mean_IC": mean_ic,
                "IC_tstat": t_stat
            }

        ic_table = pd.DataFrame(ic_results).T
        ic_table.sort_values("Mean_IC", ascending=False)

        return ic_table

    @staticmethod
    def feature_pre_selection(ic_df, cs_features, threshold=1.5, corr_threshold=0.7):

        """
        1. Filter by IC t-stat
        2. Remove highly correlated features and Keep stronger IC feature
        """

        # 1.　Filter by IC t-stat
        filtered_features = []

        for feature in ic_df.index:
            if abs(ic_df.loc[feature, "IC_tstat"]) > threshold:
                filtered_features.append(feature)

        print("After t-stat filter:", filtered_features)

        # 2. Remove highly correlated features and Keep stronger IC feature

        stacked_features = {}

        for f in filtered_features:
            stacked_features[f] = cs_features[f].stack()


        feature_df = pd.DataFrame(stacked_features)

        feature_df = feature_df.dropna()

        corr_matrix = feature_df.corr().abs()

        print("Correlation Matrix between features:", corr_matrix)

        selected = filtered_features.copy()

        for i in range(len(filtered_features)):
            for j in range(i + 1, len(filtered_features)):

                feature_1 = filtered_features[i]
                feature_2 = filtered_features[j]

                if feature_1 in selected and feature_2 in selected:
                    if corr_matrix.loc[feature_1, feature_2] > corr_threshold:
                        if abs(ic_df.loc[feature_1, "IC_tstat"]) >= abs(ic_df.loc[feature_2, "IC_tstat"]):
                            selected.remove(feature_2)
                        else:
                            selected.remove(feature_1)

        print("After correlation filter:", selected)
        return selected

    @staticmethod
    def incremental_ic_test(cs_features_train, train_panel_df, feature_list):

        print("\nComputing Incremental IC...")

        results = {}
        current_features = []

        for feature in feature_list:
            current_features.append(feature)

            # ---- Combine signals (equal weight) ----
            combined_signal = sum(cs_features_train[f] for f in current_features) / len(current_features)

            combined_signal = combined_signal.stack()
            combined_signal.name = "combined_signal"

            # Join with training panel
            temp_panel = train_panel_df.copy()
            temp_panel = temp_panel.join(combined_signal, how="inner")

            daily_ic = (
                temp_panel[["combined_signal", "target"]].groupby(level="Date")
                    .apply(lambda x: spearmanr(x["combined_signal"],x["target"])[0])
            )

            mean_ic = daily_ic.mean()
            std_ic = daily_ic.std()
            t_stat = mean_ic / std_ic * np.sqrt(len(daily_ic))

            results[" + ".join(current_features)] = {
                "Mean_IC": mean_ic,
                "IC_tstat": t_stat
            }

        return pd.DataFrame(results).T

    # IC-Decay
    @staticmethod
    def ic_decay(close,selected_cs_features,select_features,period = 31):
        decay_results = {}

        for k in range(1, period, 2):
            forward_return_k = close.pct_change(periods=k,fill_method=None).shift(-k)

            panel_k = Before_Modelling.panel_construction(selected_cs_features,forward_return_k)

            _,test_panel,_ = Before_Modelling.split_panel_data(panel_k,selected_cs_features,split_date="2023-01-01",verbose=False)

            test_panel_copy = test_panel.copy()

            test_panel_copy["signal"] = test_panel[select_features].mean(axis = 1)

            daily_ic_k = (
                test_panel_copy[["signal", "target"]]
                    .groupby(level="Date")
                    .apply(lambda x: spearmanr(
                    x["signal"], x["target"]
                )[0])
            )

            decay_results[k] = daily_ic_k.mean()

        decay_df = pd.DataFrame.from_dict(
            decay_results,
            orient="index",
            columns=["Mean_IC"]
        )

        decay_df.index.name = "Horizon"
        decay_df = decay_df.round(5)

        print(f"Decay_result:{decay_df}")
        return decay_results

    @staticmethod
    def regime_conditon_test(test_panel, spy_close, vix_close):

        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.squeeze()

        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.squeeze()

        # 20-day market return
        mar = spy_close.pct_change(20)

        # 20-day VIX z-score
        vix_mean = vix_close.rolling(20).mean()
        vix_std = vix_close.rolling(20).std()
        vix_z = (vix_close - vix_mean) / vix_std

        rolling_window = 252

        mar_upper = mar.rolling(rolling_window).quantile(0.7)
        mar_lower = mar.rolling(rolling_window).quantile(0.3)

        vix_upper = vix_z.rolling(rolling_window).quantile(0.7)
        vix_lower = vix_z.rolling(rolling_window).quantile(0.3)

        bull_regime = mar >= mar_upper
        bear_regime = mar <= mar_lower

        low_vol = vix_z <= vix_lower
        high_vol = vix_z >= vix_upper

        regime_df = pd.DataFrame({
            "bull": bull_regime,
            "bear": bear_regime,
            "low_vol": low_vol,
            "high_vol": high_vol
        })

        test_panel = test_panel.join(regime_df, on="Date")

        def compute_ic(panel):
            if panel.empty:
                return np.nan, np.nan

            daily_ic = (
                panel[["signal", "target"]]
                    .groupby(level="Date")
                    .apply(lambda x: spearmanr(
                    x["signal"], x["target"])[0])
            )

            mean_ic = daily_ic.mean()
            t_stat = mean_ic / (daily_ic.std() / np.sqrt(len(daily_ic)))

            return mean_ic, t_stat

        results = {
            "Overall": compute_ic(test_panel),
            "Bull Market": compute_ic(test_panel[test_panel["bull"]]),
            "Bear Market": compute_ic(test_panel[test_panel["bear"]]),
            "Low Vol": compute_ic(test_panel[test_panel["low_vol"]]),
            "High Vol": compute_ic(test_panel[test_panel["high_vol"]])
        }

        result_df = pd.DataFrame(results, index=["Mean_IC", "IC_tstat"]).T

        print("\n=== Regime Condition IC Test ===")
        print(result_df.round(5))

        return result_df






