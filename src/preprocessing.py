# src/preprocessing.py

import numpy as np
import pandas as pd


def clean_team_logs(df):
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return df


def check_same_day_duplicates(df):
    dup = (
        df.groupby(["TEAM_ID", "GAME_DATE"])
        .size()
        .reset_index(name="n_games")
    )
    return dup[dup["n_games"] > 1].sort_values(["TEAM_ID", "GAME_DATE"])


def engineer_rolling_features(df, window=10, min_periods=3):
    df = df.copy()
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    features = ["PACE", "OFF_RATING", "DEF_RATING"]

    for col in features:
        df[f"{col}_ROLL_MEAN_{window}"] = (
            df.groupby("TEAM_ID")[col]
            .transform(lambda x: x.shift(1).ewm(span=window, min_periods=min_periods).mean())
        )

    df["PTS_ROLL_MEAN_MU"] = (
        df.groupby("TEAM_ID")["PTS"]
        .transform(lambda x: x.shift(1).ewm(span=window, min_periods=min_periods).mean())
    )

    df["PTS_ROLL_VAR_SIGMA2"] = (
        df.groupby("TEAM_ID")["PTS"]
        .transform(lambda x: x.shift(1).ewm(span=window, min_periods=min_periods).var())
    )

    cols_needed = [f"{col}_ROLL_MEAN_{window}" for col in features] + [
        "PTS_ROLL_MEAN_MU",
        "PTS_ROLL_VAR_SIGMA2",
    ]

    df.loc[df["PTS_ROLL_VAR_SIGMA2"] <= 0, "PTS_ROLL_VAR_SIGMA2"] = np.nan
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    return df

def prepare_matchup_data(df):
    df = df.copy()

    opp_features = df[
        ["GAME_ID", "TEAM_ID", "DEF_RATING_ROLL_MEAN_10", "PACE_ROLL_MEAN_10"]
    ].copy()

    opp_features = opp_features.rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "DEF_RATING_ROLL_MEAN_10": "OPP_DEF_RATING_ROLL_MEAN_10",
        "PACE_ROLL_MEAN_10": "OPP_PACE_ROLL_MEAN_10",
    })

    merged_df = pd.merge(df, opp_features, on="GAME_ID", how="inner")
    matchup_df = merged_df[merged_df["TEAM_ID"] != merged_df["OPP_TEAM_ID"]].reset_index(drop=True)

    matchup_df = matchup_df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
    return matchup_df
