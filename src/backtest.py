import numpy as np
import pandas as pd
import scipy.stats as stats


def calc_implied_prob(odds):
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100), 100 / (odds + 100))


def calc_b_odds(odds):
    return np.where(odds < 0, 100 / np.abs(odds), odds / 100)


def execute_backtest(
    df: pd.DataFrame,
    n_col: str,
    p_col: str,
    prob_over_col: str,
    prob_under_col: str,
    bet_prefix: str = "",
    edge_threshold: float = 0.03,
    starting_bankroll: float = 10000,
) -> pd.DataFrame:
    """
    Generic backtest using model-generated probabilities.
    """
    df = df.copy()

    if "ODDS_OVER" not in df.columns:
        df["ODDS_OVER"] = -110
    if "ODDS_UNDER" not in df.columns:
        df["ODDS_UNDER"] = -110

    df["IMPLIED_PROB_OVER"] = calc_implied_prob(df["ODDS_OVER"])
    df["IMPLIED_PROB_UNDER"] = calc_implied_prob(df["ODDS_UNDER"])

    df["B_ODDS_OVER"] = calc_b_odds(df["ODDS_OVER"])
    df["B_ODDS_UNDER"] = calc_b_odds(df["ODDS_UNDER"])

    df[f"EDGE_OVER{bet_prefix}"] = df[prob_over_col] - df["IMPLIED_PROB_OVER"]
    df[f"EDGE_UNDER{bet_prefix}"] = df[prob_under_col] - df["IMPLIED_PROB_UNDER"]

    df[f"BET_PLACED{bet_prefix}"] = "NO BET"
    df[f"BET_PLACED{bet_prefix}"] = np.where(
        df[f"EDGE_OVER{bet_prefix}"] > edge_threshold,
        "OVER",
        df[f"BET_PLACED{bet_prefix}"],
    )
    df[f"BET_PLACED{bet_prefix}"] = np.where(
        (df[f"EDGE_UNDER{bet_prefix}"] > edge_threshold) & (df[f"BET_PLACED{bet_prefix}"] == "NO BET"),
        "UNDER",
        df[f"BET_PLACED{bet_prefix}"],
    )

    conditions = [
        (df[f"BET_PLACED{bet_prefix}"] == "OVER") & (df["PTS"] > df["TRUE_MARKET_LINE"]),
        (df[f"BET_PLACED{bet_prefix}"] == "UNDER") & (df["PTS"] < df["TRUE_MARKET_LINE"]),
        (df[f"BET_PLACED{bet_prefix}"] == "OVER") & (df["PTS"] < df["TRUE_MARKET_LINE"]),
        (df[f"BET_PLACED{bet_prefix}"] == "UNDER") & (df["PTS"] > df["TRUE_MARKET_LINE"]),
    ]
    df[f"BET_RESULT{bet_prefix}"] = np.select(conditions, ["WIN", "WIN", "LOSS", "LOSS"], default="PUSH")

    pnl_conditions = [df[f"BET_RESULT{bet_prefix}"] == "WIN", df[f"BET_RESULT{bet_prefix}"] == "LOSS"]
    df[f"PNL{bet_prefix}"] = np.select(pnl_conditions, [100, -110], default=0)

    df[f"CUM_PNL{bet_prefix}"] = df[f"PNL{bet_prefix}"].cumsum()
    df[f"BANKROLL{bet_prefix}"] = starting_bankroll + df[f"CUM_PNL{bet_prefix}"]

    actual_prob = stats.nbinom.pmf(k=df["PTS"], n=df[n_col], p=df[p_col])
    df[f"LOG_LIKELIHOOD{bet_prefix}"] = -np.log(actual_prob + 1e-9)

    return df
