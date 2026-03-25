from pathlib import Path

# Project root = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
TABLES_DIR = OUTPUT_DIR / "tables"

for folder in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

DEFAULT_FEATURES = [
    "OFF_RATING_ROLL_MEAN_10",
    "OPP_DEF_RATING_ROLL_MEAN_10",
    "PACE_ROLL_MEAN_10",
    "OPP_PACE_ROLL_MEAN_10",
]

TARGET_COL = "PTS"
LINE_COL = "TRUE_MARKET_LINE"
DATE_COL = "GAME_DATE"
SEASON_COL = "SEASON"
