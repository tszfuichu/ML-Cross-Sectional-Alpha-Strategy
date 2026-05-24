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
    def split_panel_data(panel_df,cs_features, split_date):

        split_date = pd.to_datetime(split_date)
        train_mask = panel_df.index.get_level_values("Date") < split_date
        test_mask = panel_df.index.get_level_values("Date") >= split_date

        train_panel = panel_df.loc[train_mask]
        test_panel = panel_df.loc[test_mask]

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




