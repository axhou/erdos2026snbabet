import pandas as pd


def add_rolling_team_features(
    df: pd.DataFrame,
    group_col: str = "TEAM_ID",
    date_col: str = "GAME_DATE",
    windows: tuple[int, ...] = (10,),
) -> pd.DataFrame:
    """
    Add rolling means and variances for selected columns.
    Assumes the columns already exist in the raw data.
    """
    df = df.copy()
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    base_cols = ["PTS", "OFF_RATING", "DEF_RATING", "PACE"]

    for w in windows:
        for col in base_cols:
            if col not in df.columns:
                continue

            roll_mean_col = f"{col}_ROLL_MEAN_{w}"
            df[roll_mean_col] = (
                df.groupby(group_col)[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
            )

        if "PTS" in df.columns:
            var_col = f"PTS_ROLL_VAR_{w}"
            df[var_col] = (
                df.groupby(group_col)["PTS"]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=3).var())
            )

    # Rename to match your notebook naming
    if "PTS_ROLL_MEAN_10" in df.columns:
        df["PTS_ROLL_MEAN_MU"] = df["PTS_ROLL_MEAN_10"]

    if "PTS_ROLL_VAR_10" in df.columns:
        df["PTS_ROLL_VAR_SIGMA2"] = df["PTS_ROLL_VAR_10"]

    return df


def prepare_matchup_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game row, attach opponent rolling defensive and pace features.
    """
    df = df.copy()

    needed_cols = [
        "GAME_ID",
        "TEAM_ID",
        "DEF_RATING_ROLL_MEAN_10",
        "PACE_ROLL_MEAN_10",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for matchup prep: {missing}")

    opp_features = df[needed_cols].copy()
    opp_features = opp_features.rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "DEF_RATING_ROLL_MEAN_10": "OPP_DEF_RATING_ROLL_MEAN_10",
            "PACE_ROLL_MEAN_10": "OPP_PACE_ROLL_MEAN_10",
        }
    )

    merged_df = pd.merge(df, opp_features, on="GAME_ID", how="left")
    matchup_df = merged_df[merged_df["TEAM_ID"] != merged_df["OPP_TEAM_ID"]].reset_index(drop=True)

    return matchup_df
