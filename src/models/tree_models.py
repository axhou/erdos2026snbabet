import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from src.config import DEFAULT_FEATURES


def train_ensemble_models(df: pd.DataFrame):
    """
    Train Random Forest and XGBoost on the same feature set.
    This is the static train-all version.
    """
    df = df.copy().dropna(
        subset=DEFAULT_FEATURES + ["PTS", "PTS_ROLL_VAR_SIGMA2", "PTS_ROLL_MEAN_MU"]
    )

    X = df[DEFAULT_FEATURES]
    y = df["PTS"]

    split_idx = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    df["RF_ADJUSTED_MU"] = rf_model.predict(X)

    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    df["XGB_ADJUSTED_MU"] = xgb_model.predict(X)

    dispersion_ratio = df["PTS_ROLL_VAR_SIGMA2"] / df["PTS_ROLL_MEAN_MU"]
    df["RF_ADJUSTED_SIGMA2"] = df["RF_ADJUSTED_MU"] * dispersion_ratio
    df["XGB_ADJUSTED_SIGMA2"] = df["XGB_ADJUSTED_MU"] * dispersion_ratio

    return df, rf_model, xgb_model


def train_xgb_walk_forward(
    df: pd.DataFrame,
    eval_season: str = "2022-23",
    min_train_rows: int = 500,
):
    """
    Walk-forward / expanding-window XGBoost evaluation.

    For each game date in eval_season:
      - train using only rows strictly earlier than that date
      - predict rows on that date

    This better matches deployment conditions and avoids look-ahead bias.
    """
    required_cols = DEFAULT_FEATURES + [
        "PTS",
        "PTS_ROLL_VAR_SIGMA2",
        "PTS_ROLL_MEAN_MU",
        "GAME_DATE",
        "SEASON",
    ]

    df = df.copy().dropna(subset=required_cols).reset_index(drop=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    sort_cols = ["GAME_DATE"]
    if "GAME_ID" in df.columns:
        sort_cols.append("GAME_ID")
    if "TEAM_ID" in df.columns:
        sort_cols.append("TEAM_ID")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    eval_df = df[df["SEASON"] == eval_season].copy()
    if eval_df.empty:
        raise ValueError(f"No rows found for eval_season={eval_season}")

    eval_dates = sorted(eval_df["GAME_DATE"].unique())
    preds = []

    print(f"Running walk-forward XGBoost on eval season {eval_season}...")
    print(f"Number of evaluation dates: {len(eval_dates)}")

    for i, current_date in enumerate(eval_dates, start=1):
        train_df = df[df["GAME_DATE"] < current_date].copy()
        test_df = eval_df[eval_df["GAME_DATE"] == current_date].copy()

        if len(train_df) < min_train_rows:
            print(
                f"Skipping {pd.Timestamp(current_date).date()} because train rows = {len(train_df)} "
                f"< min_train_rows = {min_train_rows}"
            )
            continue

        X_train = train_df[DEFAULT_FEATURES]
        y_train = train_df["PTS"]
        X_test = test_df[DEFAULT_FEATURES]

        model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )
        model.fit(X_train, y_train)

        test_df["XGB_WF_ADJUSTED_MU"] = model.predict(X_test)

        dispersion_ratio = test_df["PTS_ROLL_VAR_SIGMA2"] / test_df["PTS_ROLL_MEAN_MU"]
        test_df["XGB_WF_ADJUSTED_SIGMA2"] = test_df["XGB_WF_ADJUSTED_MU"] * dispersion_ratio

        preds.append(test_df)

        if i % 20 == 0 or i == len(eval_dates):
            print(
                f"Processed {i}/{len(eval_dates)} dates "
                f"(current date: {pd.Timestamp(current_date).date()})"
            )

    if not preds:
        raise ValueError("Walk-forward XGBoost produced no predictions.")

    pred_df = pd.concat(preds, ignore_index=True)
    return pred_df