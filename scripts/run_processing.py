import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features import add_rolling_team_features, prepare_matchup_data
from src.preprocessing import clean_team_logs

if __name__ == "__main__":
    raw_path = RAW_DATA_DIR / "nba_team_logs_raw.csv"
    df = pd.read_csv(raw_path)

    df = clean_team_logs(df)
    df = add_rolling_team_features(df, windows=(10,))
    df = prepare_matchup_data(df)

    output_path = PROCESSED_DATA_DIR / "matchup_model_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved processed matchup data to: {output_path}")
