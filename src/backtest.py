import numpy as np
import pandas as pd


def calc_implied_prob(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100), 100 / (odds + 100))


def calc_b_odds(odds):
    odds = pd.to_numeric(odds, errors="coerce")
    return np.where(odds < 0, 100 / np.abs(odds), odds / 100)


def execute_backtest(
    df: pd.DataFrame,
    prob_over_col: str,
    prob_under_col: str,
    market_line_col: str = "TRUE_MARKET_LINE",
    odds_over_col: str = "ODDS_OVER",
    odds_under_col: str = "ODDS_UNDER",
    bet_prefix: str = "",
    edge_threshold: float = 0.03,
    starting_bankroll: float = 10000.0,
) -> pd.DataFrame:
    """
    Generic backtest for over/under betting.

    If ODDS_OVER / ODDS_UNDER are missing, defaults to -110.
    """
    df = df.copy()

    if odds_over_col not in df.columns:
        df[odds_over_col] = -110
    if odds_under_col not in df.columns:
        df[odds_under_col] = -110

    df[odds_over_col] = pd.to_numeric(df[odds_over_col], errors="coerce").fillna(-110)
    df[odds_under_col] = pd.to_numeric(df[odds_under_col], errors="coerce").fillna(-110)

    df["IMPLIED_PROB_OVER"] = calc_implied_prob(df[odds_over_col])
    df["IMPLIED_PROB_UNDER"] = calc_implied_prob(df[odds_under_col])

    df["B_ODDS_OVER"] = calc_b_odds(df[odds_over_col])
    df["B_ODDS_UNDER"] = calc_b_odds(df[odds_under_col])

    df[f"EDGE_OVER{bet_prefix}"] = df[prob_over_col] - df["IMPLIED_PROB_OVER"]
    df[f"EDGE_UNDER{bet_prefix}"] = df[prob_under_col] - df["IMPLIED_PROB_UNDER"]

    df[f"BET_PLACED{bet_prefix}"] = "NO BET"

    over_mask = df[f"EDGE_OVER{bet_prefix}"] > edge_threshold
    under_mask = df[f"EDGE_UNDER{bet_prefix}"] > edge_threshold

    # If both edges exceed threshold, choose the larger edge
    both_mask = over_mask & under_mask
    over_better = df[f"EDGE_OVER{bet_prefix}"] >= df[f"EDGE_UNDER{bet_prefix}"]

    df.loc[over_mask & ~both_mask, f"BET_PLACED{bet_prefix}"] = "OVER"
    df.loc[under_mask & ~both_mask, f"BET_PLACED{bet_prefix}"] = "UNDER"
    df.loc[both_mask & over_better, f"BET_PLACED{bet_prefix}"] = "OVER"
    df.loc[both_mask & ~over_better, f"BET_PLACED{bet_prefix}"] = "UNDER"

    over_win = (df[f"BET_PLACED{bet_prefix}"] == "OVER") & (df["PTS"] > df[market_line_col])
    under_win = (df[f"BET_PLACED{bet_prefix}"] == "UNDER") & (df["PTS"] < df[market_line_col])

    over_loss = (df[f"BET_PLACED{bet_prefix}"] == "OVER") & (df["PTS"] < df[market_line_col])
    under_loss = (df[f"BET_PLACED{bet_prefix}"] == "UNDER") & (df["PTS"] > df[market_line_col])

    push_mask = (
        (df[f"BET_PLACED{bet_prefix}"].isin(["OVER", "UNDER"]))
        & (df["PTS"] == df[market_line_col])
    )

    df[f"BET_RESULT{bet_prefix}"] = "NO BET"
    df.loc[over_win | under_win, f"BET_RESULT{bet_prefix}"] = "WIN"
    df.loc[over_loss | under_loss, f"BET_RESULT{bet_prefix}"] = "LOSS"
    df.loc[push_mask, f"BET_RESULT{bet_prefix}"] = "PUSH"

    # Flat staking: risk 110 to win 100 at standard -110.
    # More generally:
    over_profit = np.where(df[odds_over_col] < 0, 100, df[odds_over_col])
    under_profit = np.where(df[odds_under_col] < 0, 100, df[odds_under_col])

    over_risk = np.where(df[odds_over_col] < 0, np.abs(df[odds_over_col]), 100)
    under_risk = np.where(df[odds_under_col] < 0, np.abs(df[odds_under_col]), 100)

    df[f"PNL{bet_prefix}"] = 0.0

    df.loc[over_win, f"PNL{bet_prefix}"] = over_profit[over_win]
    df.loc[under_win, f"PNL{bet_prefix}"] = under_profit[under_win]
    df.loc[over_loss, f"PNL{bet_prefix}"] = -over_risk[over_loss]
    df.loc[under_loss, f"PNL{bet_prefix}"] = -under_risk[under_loss]
    df.loc[push_mask, f"PNL{bet_prefix}"] = 0.0

    sort_cols = [c for c in ["GAME_DATE", "GAME_ID", "TEAM_NAME"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df[f"CUM_PNL{bet_prefix}"] = df[f"PNL{bet_prefix}"].cumsum()
    df[f"BANKROLL{bet_prefix}"] = starting_bankroll + df[f"CUM_PNL{bet_prefix}"]

    bet_count = (df[f"BET_PLACED{bet_prefix}"] != "NO BET").sum()
    win_count = (df[f"BET_RESULT{bet_prefix}"] == "WIN").sum()
    loss_count = (df[f"BET_RESULT{bet_prefix}"] == "LOSS").sum()

    print(f"Backtest summary {bet_prefix or ''}".strip())
    print(f"Total bets: {bet_count}")
    print(f"Wins: {win_count}")
    print(f"Losses: {loss_count}")
    print(f"Final bankroll: {df[f'BANKROLL{bet_prefix}'].iloc[-1]:.2f}")
    print(f"Total PnL: {df[f'CUM_PNL{bet_prefix}'].iloc[-1]:.2f}")

    return df