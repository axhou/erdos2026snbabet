import pandas as pd

from src.backtest import execute_backtest
from src.config import PREDICTIONS_DIR, TABLES_DIR

if __name__ == "__main__":
    input_path = PREDICTIONS_DIR / "tree_model_predictions.csv"
    df = pd.read_csv(input_path)

    xgb_backtest_df = execute_backtest(
        df=df,
        n_col="n_param_XGB",
        p_col="p_param_XGB",
        prob_over_col="PROB_OVER_XGB",
        prob_under_col="PROB_UNDER_XGB",
        bet_prefix="_XGB",
        edge_threshold=0.03,
        starting_bankroll=10000,
    )

    output_path = TABLES_DIR / "xgb_backtest_results.csv"
    xgb_backtest_df.to_csv(output_path, index=False)

    print(f"Saved backtest results to: {output_path}")
    print(f"Final bankroll: {xgb_backtest_df['BANKROLL_XGB'].iloc[-1]:.2f}")
    print(f"Total PnL: {xgb_backtest_df['CUM_PNL_XGB'].iloc[-1]:.2f}")
