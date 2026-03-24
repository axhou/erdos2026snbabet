from src.data_generation import fetch_nba_team_data, save_raw_team_logs

if __name__ == "__main__":
    seasons = [
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
    ]

    raw_df = fetch_nba_team_data(seasons=seasons)
    save_raw_team_logs(raw_df, filename="nba_team_logs_raw.csv")
