import time
from typing import Iterable

import pandas as pd
from nba_api.stats.endpoints import teamgamelogs
from requests.exceptions import ReadTimeout, ConnectionError

from src.config import RAW_DATA_DIR


def fetch_nba_team_data(seasons: Iterable[str], max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch NBA team game logs from nba_api for multiple seasons.

    Returns
    -------
    pd.DataFrame
        Combined raw dataframe across seasons.
    """
    appended_data = []

    for season in seasons:
        print(f"Fetching NBA API data for season: {season}...")
        advanced_logs = None
        basic_logs = None

        for attempt in range(max_retries):
            try:
                advanced_logs = teamgamelogs.TeamGameLogs(
                    season_nullable=season,
                    measure_type_player_game_logs_nullable="Advanced",
                    timeout=60,
                ).get_data_frames()[0]

                basic_logs = teamgamelogs.TeamGameLogs(
                    season_nullable=season,
                    measure_type_player_game_logs_nullable="Base",
                    timeout=60,
                ).get_data_frames()[0]
                break

            except (ReadTimeout, ConnectionError):
                print(
                    f"[!] Timeout on {season}, attempt {attempt + 1}/{max_retries}. "
                    "Retrying in 5 seconds..."
                )
                time.sleep(5)

        if advanced_logs is None or basic_logs is None:
            print(f"[!] Skipping season {season} due to repeated API failures.")
            continue

        advanced_logs = advanced_logs.drop(columns=["TEAM_NAME", "MATCHUP"], errors="ignore")

        cols_to_merge = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "MATCHUP", "PTS"]
        merged_df = pd.merge(
            advanced_logs,
            basic_logs[cols_to_merge],
            on=["GAME_ID", "TEAM_ID"],
            how="left",
        )

        merged_df["SEASON"] = season
        appended_data.append(merged_df)

        time.sleep(1)

    if not appended_data:
        raise ValueError("No data was fetched from nba_api.")

    full_df = pd.concat(appended_data, ignore_index=True)
    return full_df


def save_raw_team_logs(df: pd.DataFrame, filename: str = "nba_team_logs_raw.csv") -> None:
    output_path = RAW_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved raw team logs to: {output_path}")
