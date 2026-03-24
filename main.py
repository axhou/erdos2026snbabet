from pathlib import Path
import argparse
import pandas as pd

from src.data_generation import fetch_nba_team_data, scrape_multi_year_sbr_odds, save_dataframe
from src.preprocessing import clean_team_logs, engineer_rolling_features, prepare_matchup_data


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_generate_data():
    ensure_dirs()

    seasons = [
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
    ]
    start_years = [2018, 2019, 2020, 2021, 2022]

    raw_logs = fetch_nba_team_data(seasons)
    save_dataframe(raw_logs, RAW_DIR / "nba_team_logs_raw.csv")

    raw_odds = scrape_multi_year_sbr_odds(start_years)
    save_dataframe(raw_odds, RAW_DIR / "sbr_odds_raw.csv")

    print("Raw data saved.")


def run_preprocess():
    ensure_dirs()

    nba_df = pd.read_csv(RAW_DIR / "nba_team_logs_raw.csv")
    odds_df = pd.read_csv(RAW_DIR / "sbr_odds_raw.csv")

    nba_df = clean_team_logs(nba_df)

    merged_df = nba_df.copy()

    merged_df = engineer_rolling_features(merged_df)
    matchup_df = prepare_matchup_data(merged_df)

    matchup_df.to_csv(PROCESSED_DIR / "nba_matchups_features.csv", index=False)
    print("Processed feature dataset saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["generate-data", "preprocess"])
    args = parser.parse_args()

    if args.step == "generate-data":
        run_generate_data()
    elif args.step == "preprocess":
        run_preprocess()


if __name__ == "__main__":
    main()
