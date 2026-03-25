from pathlib import Path
import argparse
import pandas as pd

from src.data_generation import fetch_nba_team_data, scrape_multi_year_sbr_odds
from src.preprocessing import clean_team_logs
from src.features import add_rolling_team_features, prepare_matchup_data
from src.models.glm_model import fit_glm_and_predict
from src.models.tree_models import train_ensemble_models, train_xgb_walk_forward
from src.evaluation import evaluate_glm, evaluate_ensemble_nll, evaluate_xgb_walk_forward
from src.backtest import execute_backtest

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PREDICTIONS_DIR = Path("outputs/predictions")
TABLES_DIR = Path("outputs/tables")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


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
    raw_logs.to_csv(RAW_DIR / "nba_team_logs_raw.csv", index=False)
    print(f"Saved raw logs to {RAW_DIR / 'nba_team_logs_raw.csv'}")

    raw_odds = scrape_multi_year_sbr_odds(start_years)
    raw_odds.to_csv(RAW_DIR / "sbr_odds_raw.csv", index=False)
    print(f"Saved SBR odds to {RAW_DIR / 'sbr_odds_raw.csv'}")


def run_preprocess():
    ensure_dirs()

    nba_df = pd.read_csv(RAW_DIR / "nba_team_logs_raw.csv")
    odds_df = pd.read_csv(RAW_DIR / "sbr_odds_raw.csv")

    nba_df = clean_team_logs(nba_df)
    nba_df["GAME_DATE"] = pd.to_datetime(nba_df["GAME_DATE"])
    odds_df["GAME_DATE"] = pd.to_datetime(odds_df["GAME_DATE"])

    feature_df = add_rolling_team_features(nba_df, window=10)
    feature_df = pd.merge(
        feature_df,
        odds_df,
        on=["GAME_DATE", "TEAM_NAME"],
        how="inner",
    )

    matchup_df = prepare_matchup_data(feature_df, window=10)

    matchup_df.to_csv(PROCESSED_DIR / "nba_matchups_features.csv", index=False)
    print(f"Processed feature dataset saved to {PROCESSED_DIR / 'nba_matchups_features.csv'}")


def run_glm():
    ensure_dirs()

    df = pd.read_csv(PROCESSED_DIR / "nba_matchups_features.csv")
    glm_df, glm_model = fit_glm_and_predict(df)
    glm_df = evaluate_glm(glm_df)

    out_path = PREDICTIONS_DIR / "glm_predictions.csv"
    glm_df.to_csv(out_path, index=False)

    print(f"Saved GLM predictions to {out_path}")
    print(f"Mean GLM NLL: {glm_df['LOG_LIKELIHOOD_GLM'].mean():.4f}")


def run_tree_models():
    ensure_dirs()

    df = pd.read_csv(PROCESSED_DIR / "nba_matchups_features.csv")
    tree_df, rf_model, xgb_model = train_ensemble_models(df)
    tree_df = evaluate_ensemble_nll(tree_df)

    out_path = PREDICTIONS_DIR / "tree_model_predictions.csv"
    tree_df.to_csv(out_path, index=False)

    print(f"Saved tree model predictions to {out_path}")
    print(f"Mean RF NLL: {tree_df['LOG_LIKELIHOOD_RF'].mean():.4f}")
    print(f"Mean XGB NLL: {tree_df['LOG_LIKELIHOOD_XGB'].mean():.4f}")


def run_xgb_walk_forward(eval_season: str):
    ensure_dirs()

    df = pd.read_csv(PROCESSED_DIR / "nba_matchups_features.csv")
    wf_df = train_xgb_walk_forward(df, eval_season=eval_season)
    wf_df = evaluate_xgb_walk_forward(wf_df)

    out_path = PREDICTIONS_DIR / f"xgb_walkforward_predictions_{eval_season}.csv"
    wf_df.to_csv(out_path, index=False)

    print(f"Saved walk-forward XGBoost predictions to {out_path}")
    print(f"Mean XGB walk-forward NLL: {wf_df['LOG_LIKELIHOOD_XGB_WF'].mean():.6f}")


def run_backtest(model: str, edge_threshold: float, eval_season: str):
    ensure_dirs()

    if model == "glm":
        df = pd.read_csv(PREDICTIONS_DIR / "glm_predictions.csv")
        result_df = execute_backtest(
            df=df,
            prob_over_col="PROB_OVER_GLM",
            prob_under_col="PROB_UNDER_GLM",
            market_line_col="TRUE_MARKET_LINE",
            bet_prefix="_GLM",
            edge_threshold=edge_threshold,
            starting_bankroll=10000,
        )
        out_path = TABLES_DIR / "glm_backtest_results.csv"

    elif model == "rf":
        df = pd.read_csv(PREDICTIONS_DIR / "tree_model_predictions.csv")
        result_df = execute_backtest(
            df=df,
            prob_over_col="PROB_OVER_RF",
            prob_under_col="PROB_UNDER_RF",
            market_line_col="TRUE_MARKET_LINE",
            bet_prefix="_RF",
            edge_threshold=edge_threshold,
            starting_bankroll=10000,
        )
        out_path = TABLES_DIR / "rf_backtest_results.csv"

    elif model == "xgb":
        df = pd.read_csv(PREDICTIONS_DIR / "tree_model_predictions.csv")
        result_df = execute_backtest(
            df=df,
            prob_over_col="PROB_OVER_XGB",
            prob_under_col="PROB_UNDER_XGB",
            market_line_col="TRUE_MARKET_LINE",
            bet_prefix="_XGB",
            edge_threshold=edge_threshold,
            starting_bankroll=10000,
        )
        out_path = TABLES_DIR / "xgb_backtest_results.csv"

    elif model == "xgb-wf":
        df = pd.read_csv(PREDICTIONS_DIR / f"xgb_walkforward_predictions_{eval_season}.csv")
        result_df = execute_backtest(
            df=df,
            prob_over_col="PROB_OVER_XGB_WF",
            prob_under_col="PROB_UNDER_XGB_WF",
            market_line_col="TRUE_MARKET_LINE",
            bet_prefix="_XGB_WF",
            edge_threshold=edge_threshold,
            starting_bankroll=10000,
        )
        out_path = TABLES_DIR / f"xgb_walkforward_backtest_results_{eval_season}.csv"

    else:
        raise ValueError("model must be one of: glm, rf, xgb, xgb-wf")

    result_df.to_csv(out_path, index=False)
    print(f"Saved backtest results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="step", required=True)

    subparsers.add_parser("generate-data")
    subparsers.add_parser("preprocess")
    subparsers.add_parser("glm")
    subparsers.add_parser("tree-models")

    wf_parser = subparsers.add_parser("xgb-walk-forward")
    wf_parser.add_argument("--eval-season", default="2022-23")

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--model", choices=["glm", "rf", "xgb", "xgb-wf"], default="glm")
    backtest_parser.add_argument("--edge-threshold", type=float, default=0.03)
    backtest_parser.add_argument("--eval-season", default="2022-23")

    args = parser.parse_args()

    if args.step == "generate-data":
        run_generate_data()
    elif args.step == "preprocess":
        run_preprocess()
    elif args.step == "glm":
        run_glm()
    elif args.step == "tree-models":
        run_tree_models()
    elif args.step == "xgb-walk-forward":
        run_xgb_walk_forward(eval_season=args.eval_season)
    elif args.step == "backtest":
        run_backtest(
            model=args.model,
            edge_threshold=args.edge_threshold,
            eval_season=args.eval_season,
        )


if __name__ == "__main__":
    main()