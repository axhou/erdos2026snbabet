# src/data_generation.py

import io
import time
import requests
import pandas as pd
from nba_api.stats.endpoints import teamgamelogs
from requests.exceptions import ReadTimeout, ConnectionError


def fetch_nba_team_data(seasons, max_retries=3, sleep_seconds=2.5):
    appended_data = []

    for season in seasons:
        print(f"Fetching NBA API data for season: {season}...")
        advanced_logs, basic_logs = None, None

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
                print(f"[!] Timeout on attempt {attempt + 1}/{max_retries}; retrying...")
                time.sleep(5)

        if advanced_logs is None or basic_logs is None:
            print(f"[!] Skipping season {season} after repeated failures.")
            continue

        advanced_logs = advanced_logs.drop(columns=["TEAM_NAME", "MATCHUP"], errors="ignore")
        cols_to_merge = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "MATCHUP", "PTS"]

        merged_df = pd.merge(
            advanced_logs,
            basic_logs[cols_to_merge],
            on=["GAME_ID", "TEAM_ID"],
            how="left",
        )

        merged_df = merged_df[merged_df["TEAM_ID"].between(1610612737, 1610612766)]
        appended_data.append(merged_df)

        time.sleep(sleep_seconds)

    if not appended_data:
        raise ValueError("No NBA data fetched.")

    raw_df = pd.concat(appended_data, ignore_index=True)
    raw_df["GAME_DATE"] = pd.to_datetime(raw_df["GAME_DATE"])

    raw_df["SEASON"] = raw_df["GAME_DATE"].apply(
        lambda d: f"{d.year}-{str(d.year + 1)[-2:]}" if d.month >= 10 else f"{d.year - 1}-{str(d.year)[-2:]}"
    )

    raw_df = raw_df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return raw_df


def save_dataframe(df, path):
    df.to_csv(path, index=False)


# src/data_generation.py （接上）

def scrape_multi_year_sbr_odds(start_years):
    all_odds = []
    headers = {"User-Agent": "Mozilla/5.0"}

    team_mapping = {
        "Atlanta": "Atlanta Hawks",
        "Boston": "Boston Celtics",
        "Brooklyn": "Brooklyn Nets",
        "Charlotte": "Charlotte Hornets",
        "Chicago": "Chicago Bulls",
        "Cleveland": "Cleveland Cavaliers",
        "Dallas": "Dallas Mavericks",
        "Denver": "Denver Nuggets",
        "Detroit": "Detroit Pistons",
        "GoldenState": "Golden State Warriors",
        "Houston": "Houston Rockets",
        "Indiana": "Indiana Pacers",
        "LAClippers": "LA Clippers",
        "LALakers": "Los Angeles Lakers",
        "Memphis": "Memphis Grizzlies",
        "Miami": "Miami Heat",
        "Milwaukee": "Milwaukee Bucks",
        "Minnesota": "Minnesota Timberwolves",
        "NewOrleans": "New Orleans Pelicans",
        "NewYork": "New York Knicks",
        "OklahomaCity": "Oklahoma City Thunder",
        "Orlando": "Orlando Magic",
        "Philadelphia": "Philadelphia 76ers",
        "Phoenix": "Phoenix Suns",
        "Portland": "Portland Trail Blazers",
        "Sacramento": "Sacramento Kings",
        "SanAntonio": "San Antonio Spurs",
        "Toronto": "Toronto Raptors",
        "Utah": "Utah Jazz",
        "Washington": "Washington Wizards",
    }

    for year in start_years:
        season_str = f"{year}-{str(year + 1)[-2:]}"
        print(f"Downloading SBR odds for {season_str}...")
        url = f"https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season_str}/"

        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        odds_df = pd.read_html(io.StringIO(response.text), header=0)[0]

        # 你原 notebook 里后续的 parsing 逻辑原样搬过来
        # ...
        # 最后 append 到 all_odds

    final_odds_df = pd.concat(all_odds, ignore_index=True)
    return final_odds_df
