import pandas as pd
import numpy as np
import time
import requests
import io

from nba_api.stats.endpoints import teamgamelogs
from requests.exceptions import ReadTimeout, ConnectionError


def fetch_nba_team_data(seasons, max_retries=3):
    """
    Fetch NBA team game logs from nba_api for multiple seasons.
    """
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
                print(
                    f"  [!] Connection timed out on attempt {attempt + 1}/{max_retries}. "
                    "Retrying in 5 seconds..."
                )
                time.sleep(5)

        if advanced_logs is None or basic_logs is None:
            print(f"  [!] Skipping season {season} due to repeated API failures.")
            continue

        advanced_logs = advanced_logs.drop(columns=["TEAM_NAME", "MATCHUP"], errors="ignore")
        cols_to_merge = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "MATCHUP", "PTS"]
        merged_df = pd.merge(
            advanced_logs,
            basic_logs[cols_to_merge],
            on=["GAME_ID", "TEAM_ID"],
            how="left",
        )

        # Keep only the 30 standard NBA teams
        merged_df = merged_df[merged_df["TEAM_ID"].between(1610612737, 1610612766)]

        appended_data.append(merged_df)
        time.sleep(2.5)

    if not appended_data:
        raise ValueError(
            "No data was fetched. The NBA API may be temporarily blocking your IP. Try again later."
        )

    raw_df = pd.concat(appended_data, ignore_index=True)
    raw_df["GAME_DATE"] = pd.to_datetime(raw_df["GAME_DATE"])

    raw_df["SEASON"] = raw_df["GAME_DATE"].apply(
        lambda d: f"{d.year}-{str(d.year + 1)[-2:]}" if d.month >= 10 else f"{d.year - 1}-{str(d.year)[-2:]}"
    )

    return raw_df


def scrape_multi_year_sbr_odds(start_years):
    """
    Parse SBR archive pages and construct team-level TRUE_MARKET_LINE.
    Raw SBR structure:
    - each game appears as two consecutive rows
    - one Close value is the game total
    - the other Close value is the spread
    """
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
        print(f"Downloading SBR Odds for {season_str}...")
        url = f"https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season_str}/"

        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            odds_df = pd.read_html(io.StringIO(response.text), header=0)[0]

            if "Date" not in odds_df.columns:
                odds_df.columns = odds_df.iloc[0]
                odds_df = odds_df[1:].reset_index(drop=True)

            odds_df = odds_df.dropna(subset=["Date"])
            odds_df = odds_df[odds_df["Date"] != "Date"].copy()

            def fix_sbr_date(date_int):
                date_str = str(int(float(date_int))).zfill(4)
                month = int(date_str[:2])
                y = year if month >= 10 else year + 1
                return pd.to_datetime(f"{y}-{date_str[:2]}-{date_str[2:]}")

            odds_df["GAME_DATE"] = odds_df["Date"].apply(fix_sbr_date)

            odds_df = odds_df[["GAME_DATE", "VH", "Team", "Close"]].copy()
            odds_df["Close"] = odds_df["Close"].replace(
                {"pk": 0, "PK": 0, "Pk": 0, "NL": np.nan, "nl": np.nan}
            )
            odds_df["Close"] = pd.to_numeric(odds_df["Close"], errors="coerce")
            odds_df = odds_df.dropna(subset=["GAME_DATE", "VH", "Team", "Close"]).reset_index(drop=True)

            parsed_games = []

            for i in range(0, len(odds_df) - 1, 2):
                pair = odds_df.iloc[i:i+2].copy()

                if len(pair) != 2:
                    continue
                if pair["GAME_DATE"].nunique() != 1:
                    continue
                if pair["Team"].nunique() != 2:
                    continue
                if set(pair["VH"]) != {"V", "H"}:
                    continue

                closes = pair["Close"].tolist()

                total_candidates = [x for x in closes if x >= 100]
                spread_candidates = [x for x in closes if abs(x) < 30]

                if len(total_candidates) != 1 or len(spread_candidates) != 1:
                    continue

                game_total = float(total_candidates[0])
                spread = float(spread_candidates[0])

                pair["IS_FAVORITE"] = np.isclose(pair["Close"], spread)

                pair["TRUE_MARKET_LINE"] = np.where(
                    pair["IS_FAVORITE"],
                    game_total / 2 + abs(spread) / 2,
                    game_total / 2 - abs(spread) / 2,
                )

                parsed_games.append(pair)

            if not parsed_games:
                print(f"Failed to parse any valid SBR games for {season_str}")
                continue

            odds_clean = pd.concat(parsed_games, ignore_index=True)
            odds_clean["TEAM_NAME"] = odds_clean["Team"].map(team_mapping)
            odds_clean = odds_clean.dropna(subset=["TEAM_NAME", "TRUE_MARKET_LINE"])

            print(f"{season_str}: parsed {len(odds_clean)} team-rows")

            all_odds.append(
                odds_clean[["GAME_DATE", "TEAM_NAME", "TRUE_MARKET_LINE"]]
            )

        except Exception as e:
            print(f"Failed to scrape {season_str}: {e}")

    if not all_odds:
        raise ValueError("No SBR odds were successfully parsed.")

    return pd.concat(all_odds, ignore_index=True)