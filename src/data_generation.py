import pandas as pd
import numpy as np
import time
import requests
import io
from nba_api.stats.endpoints import teamgamelogs
from requests.exceptions import ReadTimeout, ConnectionError

def fetch_nba_team_data(seasons, max_retries=3):
  #The nba api is weird, we have to try multiple times to get it
    appended_data = []
    for season in seasons:
        print(f"Fetching NBA API data for season: {season}...")
        advanced_logs, basic_logs = None, None
        for attempt in range(max_retries):
            try:
                advanced_logs = teamgamelogs.TeamGameLogs(
                    season_nullable=season, measure_type_player_game_logs_nullable='Advanced', timeout=60
                ).get_data_frames()[0]
                basic_logs = teamgamelogs.TeamGameLogs(
                    season_nullable=season, measure_type_player_game_logs_nullable='Base', timeout=60
                ).get_data_frames()[0]
                break
            except (ReadTimeout, ConnectionError):
                print(f"  [!] Connection timed out on attempt {attempt + 1}/{max_retries}. Retrying in 5 seconds...")
                time.sleep(5)

        if advanced_logs is None or basic_logs is None:
            print(f"  [!] Skipping season {season} due to repeated API failures.")
            continue

        advanced_logs = advanced_logs.drop(columns=['TEAM_NAME', 'MATCHUP'], errors='ignore')
        cols_to_merge = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'MATCHUP', 'PTS']
        merged_df = pd.merge(advanced_logs, basic_logs[cols_to_merge], on=['GAME_ID', 'TEAM_ID'], how='left')

        # Keep ONLY the 30 standard NBA teams to remove All-Star/Exhibition games
        merged_df = merged_df[merged_df['TEAM_ID'].between(1610612737, 1610612766)]
        appended_data.append(merged_df)
        time.sleep(2.5)

    if not appended_data:
        raise ValueError("No data was fetched. The NBA API is temporarily blocking your IP. Try again later")

    raw_df = pd.concat(appended_data, ignore_index=True)
    raw_df['GAME_DATE'] = pd.to_datetime(raw_df['GAME_DATE'])

    raw_df['SEASON'] = raw_df['GAME_DATE'].apply(
        lambda d: f"{d.year}-{str(d.year + 1)[-2:]}" if d.month >= 10 else f"{d.year - 1}-{str(d.year)[-2:]}"
    )
    return raw_df

def engineer_rolling_features(df, window=10):
    df = df.copy()
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    features = ['PACE', 'OFF_RATING', 'DEF_RATING']

    for col in features:
        df[f'{col}_ROLL_MEAN_{window}'] = df.groupby('TEAM_ID')[col].transform(
            lambda x: x.shift(1).ewm(span=window, min_periods=3).mean()
        )

    df['PTS_ROLL_MEAN_MU'] = df.groupby('TEAM_ID')['PTS'].transform(
        lambda x: x.shift(1).ewm(span=window, min_periods=3).mean()
    )
    df['PTS_ROLL_VAR_SIGMA2'] = df.groupby('TEAM_ID')['PTS'].transform(
        lambda x: x.shift(1).ewm(span=window, min_periods=3).var()
    )

    cols_needed = [f'{col}_ROLL_MEAN_{window}' for col in features] + [
        'PTS_ROLL_MEAN_MU',
        'PTS_ROLL_VAR_SIGMA2'
    ]
    df.loc[df['PTS_ROLL_VAR_SIGMA2'] <= 0, 'PTS_ROLL_VAR_SIGMA2'] = np.nan

    return df.dropna(subset=cols_needed).reset_index(drop=True)

def scrape_multi_year_sbr_odds(start_years):
    """
    Raw SBR structure:
    - each game appears as two consecutive rows
    - one Close value is the game total
    - the other Close value is the spread
    """
    all_odds = []
    headers = {"User-Agent": "Mozilla/5.0"}

    team_mapping = {
        'Atlanta': 'Atlanta Hawks', 'Boston': 'Boston Celtics', 'Brooklyn': 'Brooklyn Nets',
        'Charlotte': 'Charlotte Hornets', 'Chicago': 'Chicago Bulls', 'Cleveland': 'Cleveland Cavaliers',
        'Dallas': 'Dallas Mavericks', 'Denver': 'Denver Nuggets', 'Detroit': 'Detroit Pistons',
        'GoldenState': 'Golden State Warriors', 'Houston': 'Houston Rockets', 'Indiana': 'Indiana Pacers',
        'LAClippers': 'LA Clippers', 'LALakers': 'Los Angeles Lakers', 'Memphis': 'Memphis Grizzlies',
        'Miami': 'Miami Heat', 'Milwaukee': 'Milwaukee Bucks', 'Minnesota': 'Minnesota Timberwolves',
        'NewOrleans': 'New Orleans Pelicans', 'NewYork': 'New York Knicks', 'OklahomaCity': 'Oklahoma City Thunder',
        'Orlando': 'Orlando Magic', 'Philadelphia': 'Philadelphia 76ers', 'Phoenix': 'Phoenix Suns',
        'Portland': 'Portland Trail Blazers', 'Sacramento': 'Sacramento Kings', 'SanAntonio': 'San Antonio Spurs',
        'Toronto': 'Toronto Raptors', 'Utah': 'Utah Jazz', 'Washington': 'Washington Wizards'
    }

    for year in start_years:
        season_str = f"{year}-{str(year + 1)[-2:]}"
        print(f"Downloading SBR Odds for {season_str}...")
        url = f"https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season_str}/"

        try:
            response = requests.get(url, headers=headers)
            odds_df = pd.read_html(io.StringIO(response.text), header=0)[0]

            if 'Date' not in odds_df.columns:
                odds_df.columns = odds_df.iloc[0]
                odds_df = odds_df[1:].reset_index(drop=True)

            odds_df = odds_df.dropna(subset=['Date'])
            odds_df = odds_df[odds_df['Date'] != 'Date'].copy()

            def fix_sbr_date(date_int):
                date_str = str(int(float(date_int))).zfill(4)
                month = int(date_str[:2])
                y = year if month >= 10 else year + 1
                return pd.to_datetime(f"{y}-{date_str[:2]}-{date_str[2:]}")

            odds_df['GAME_DATE'] = odds_df['Date'].apply(fix_sbr_date)

            # We only keep the fields needed for constructing market-implied team totals
            odds_df = odds_df[['GAME_DATE', 'VH', 'Team', 'Close']].copy()
            odds_df['Close'] = odds_df['Close'].replace({
                'pk': 0, 'PK': 0, 'Pk': 0,
                'NL': np.nan, 'nl': np.nan
            })
            odds_df['Close'] = pd.to_numeric(odds_df['Close'], errors='coerce')

            odds_df = odds_df.dropna(subset=['GAME_DATE', 'VH', 'Team', 'Close']).reset_index(drop=True)

            parsed_games = []

            for i in range(0, len(odds_df) - 1, 2):
                pair = odds_df.iloc[i:i+2].copy()

                if len(pair) != 2:
                    continue

                # sanity checks
                if pair['GAME_DATE'].nunique() != 1:
                    continue
                if pair['Team'].nunique() != 2:
                    continue
                if set(pair['VH']) != {'V', 'H'}:
                    continue

                closes = pair['Close'].tolist()

                total_candidates = [x for x in closes if x >= 100]
                spread_candidates = [x for x in closes if abs(x) < 30]

                if len(total_candidates) != 1 or len(spread_candidates) != 1:
                    continue

                game_total = float(total_candidates[0])
                spread = float(spread_candidates[0])

                pair['IS_FAVORITE'] = np.isclose(pair['Close'], spread)

                pair['TRUE_MARKET_LINE'] = np.where(
                    pair['IS_FAVORITE'],
                    game_total / 2 + abs(spread) / 2,
                    game_total / 2 - abs(spread) / 2
                )

                parsed_games.append(pair)

            if not parsed_games:
                print(f"Failed to parse any valid SBR games for {season_str}")
                continue

            odds_clean = pd.concat(parsed_games, ignore_index=True)
            odds_clean['TEAM_NAME'] = odds_clean['Team'].map(team_mapping)

            odds_clean = odds_clean.dropna(subset=['TEAM_NAME', 'TRUE_MARKET_LINE'])

            print(f"{season_str}: parsed {len(odds_clean)} team-rows")
            print(odds_clean[['GAME_DATE', 'VH', 'Team', 'Close', 'IS_FAVORITE', 'TRUE_MARKET_LINE']].head(6))

            all_odds.append(
                odds_clean[['GAME_DATE', 'TEAM_NAME', 'TRUE_MARKET_LINE']]
            )

        except Exception as e:
            print(f"Failed to scrape {season_str}: {e}")

    if not all_odds:
        raise ValueError("No SBR odds were successfully parsed.")

    return pd.concat(all_odds, ignore_index=True)

# ==========================================
if __name__ == "__main__":
    sbr_data = scrape_multi_year_sbr_odds(start_years=[2019, 2020, 2021, 2022])
    seasons_to_pull = ['2019-20', '2020-21', '2021-22', '2022-23']
    raw_nba_data = fetch_nba_team_data(seasons_to_pull)
    engineered_df = engineer_rolling_features(raw_nba_data, window=10)
    engineered_df = pd.merge(engineered_df, sbr_data, on=['GAME_DATE', 'TEAM_NAME'], how='inner')

    print("\n--- Four-Season Data Ingestion Complete ---")
    print(f"Total Games in Dataset: {len(engineered_df)}")
    print(engineered_df[['TEAM_NAME', 'GAME_DATE', 'SEASON', 'PTS_ROLL_MEAN_MU', 'TRUE_MARKET_LINE']].tail())
