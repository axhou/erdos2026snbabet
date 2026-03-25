import pandas as pd
import numpy as np


def add_rolling_team_features(df: pd.DataFrame, window: int = 10, windows=None) -> pd.DataFrame:
    """
    Exponentially weighted rolling features, aligned with the original notebook logic.
    Accepts either window=10 or windows=(10,) for compatibility.
    """
    if windows is not None:
        if isinstance(windows, (tuple, list)) and len(windows) > 0:
            window = windows[0]

    df = df.copy()

    sort_cols = ["TEAM_ID", "GAME_DATE"]
    if "GAME_ID" in df.columns:
        sort_cols.append("GAME_ID")
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    features = ["PACE", "OFF_RATING", "DEF_RATING"]

    for col in features:
        df[f"{col}_ROLL_MEAN_{window}"] = df.groupby("TEAM_ID")[col].transform(
            lambda x: x.shift(1).ewm(span=window, min_periods=3).mean()
        )

    df["PTS_ROLL_MEAN_MU"] = df.groupby("TEAM_ID")["PTS"].transform(
        lambda x: x.shift(1).ewm(span=window, min_periods=3).mean()
    )

    df["PTS_ROLL_VAR_SIGMA2"] = df.groupby("TEAM_ID")["PTS"].transform(
        lambda x: x.shift(1).ewm(span=window, min_periods=3).var()
    )

    cols_needed = [f"{col}_ROLL_MEAN_{window}" for col in features] + [
        "PTS_ROLL_MEAN_MU",
        "PTS_ROLL_VAR_SIGMA2",
    ]

    df.loc[df["PTS_ROLL_VAR_SIGMA2"] <= 0, "PTS_ROLL_VAR_SIGMA2"] = np.nan

    return df.dropna(subset=cols_needed).reset_index(drop=True)


def prepare_matchup_data(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Attach opponent rolling defensive and pace features at the matchup level.
    """
    df = df.copy()

    needed_cols = [
        "GAME_ID",
        "TEAM_ID",
        f"DEF_RATING_ROLL_MEAN_{window}",
        f"PACE_ROLL_MEAN_{window}",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for matchup prep: {missing}")

    opp_features = df[needed_cols].copy()
    opp_features = opp_features.rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            f"DEF_RATING_ROLL_MEAN_{window}": "OPP_DEF_RATING_ROLL_MEAN_10",
            f"PACE_ROLL_MEAN_{window}": "OPP_PACE_ROLL_MEAN_10",
        }
    )

    merged_df = pd.merge(df, opp_features, on="GAME_ID", how="left")
    matchup_df = merged_df[merged_df["TEAM_ID"] != merged_df["OPP_TEAM_ID"]].reset_index(drop=True)

    return matchup_df