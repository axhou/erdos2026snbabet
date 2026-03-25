import pandas as pd


def clean_team_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for raw NBA team logs.
    """
    df = df.copy()

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].replace([float("inf"), -float("inf")], pd.NA)

    return df


def merge_market_lines(
    games_df: pd.DataFrame,
    market_df: pd.DataFrame,
    on_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge games data with market lines.
    Modify `on_cols` depending on your actual raw market data schema.
    """
    if on_cols is None:
        on_cols = ["GAME_ID", "TEAM_ID"]

    merged = pd.merge(games_df, market_df, on=on_cols, how="left")
    return merged


def save_processed_data(df: pd.DataFrame, output_path) -> None:
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")

def check_same_day_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether any team has multiple games on the same calendar date.
    Returns the duplicated team-date groups.
    """
    tmp = df.copy()
    if "GAME_DATE" in tmp.columns:
        tmp["GAME_DATE"] = pd.to_datetime(tmp["GAME_DATE"])

    dup = (
        tmp.groupby(["TEAM_ID", "GAME_DATE"])
        .size()
        .reset_index(name="n")
        .query("n > 1")
        .sort_values(["TEAM_ID", "GAME_DATE"])
    )

    if len(dup) > 0:
        print("[Warning] Found team-date duplicates:")
        print(dup.head(20))
    else:
        print("No same-team multiple-game same-date duplicates found.")

    return dup