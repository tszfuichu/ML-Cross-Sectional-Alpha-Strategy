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


