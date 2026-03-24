# src/backtest.py

import numpy as np
import pandas as pd


def american_odds_to_implied_prob(odds):
    odds = np.asarray(odds)
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100), 100 / (odds + 100))


def american_odds_to_b(odds):
    odds = np.asarray(odds)
    return np.where(odds < 0, 100 / np.abs(odds), odds / 100)


def run_backtest(
    df,
    prob_over_col,
    prob_under_col,
    line_col="TRUE_MARKET_LINE",
    actual_col="PTS",
    odds_over_col="ODDS_OVER",
    odds_under_col="ODDS_UNDER",
    edge_threshold=0.03,
    prefix="",
):
    df = df.copy()

    if odds_over_col not in df.columns:
        df[odds_over_col] = -110
    if odds_under_col not in df.columns:
        df[odds_under_col] = -110

    df["IMPLIED_PROB_OVER"] = american_odds_to_implied_prob(df[odds_over_col])
    df["IMPLIED_PROB_UNDER"] = american_odds_to_implied_prob(df[odds_under_col])

    edge_over_col = f"EDGE_OVER_{prefix}" if prefix else "EDGE_OVER"
    edge_under_col = f"EDGE_UNDER_{prefix}" if prefix else "EDGE_UNDER"
    bet_col = f"BET_PLACED_{prefix}" if prefix else "BET_PLACED"
    result_col = f"BET_RESULT_{prefix}" if prefix else "BET_RESULT"
    pnl_col = f"PNL_{prefix}" if prefix else "PNL"

    df[edge_over_col] = df[prob_over_col] - df["IMPLIED_PROB_OVER"]
    df[edge_under_col] = df[prob_under_col] - df["IMPLIED_PROB_UNDER"]

    df[bet_col] = "NO BET"
    df[bet_col] = np.where(df[edge_over_col] > edge_threshold, "OVER", df[bet_col])
    df[bet_col] = np.where(
        (df[edge_under_col] > edge_threshold) & (df[bet_col] == "NO BET"),
        "UNDER",
        df[bet_col],
    )

    conditions = [
        (df[bet_col] == "OVER") & (df[actual_col] > df[line_col]),
        (df[bet_col] == "UNDER") & (df[actual_col] < df[line_col]),
        (df[bet_col] == "OVER") & (df[actual_col] < df[line_col]),
        (df[bet_col] == "UNDER") & (df[actual_col] > df[line_col]),
    ]
    df[result_col] = np.select(conditions, ["WIN", "WIN", "LOSS", "LOSS"], default="PUSH")

    # baseline fixed stake
    pnl_conditions = [df[result_col] == "WIN", df[result_col] == "LOSS"]
    df[pnl_col] = np.select(pnl_conditions, [100, -110], default=0)

    return df

def run_kelly_backtest(
    df,
    prob_over_col,
    prob_under_col,
    line_col="TRUE_MARKET_LINE",
    actual_col="PTS",
    odds_over_col="ODDS_OVER",
    odds_under_col="ODDS_UNDER",
    edge_threshold=0.03,
    starting_bankroll=10000,
    kelly_fraction_scale=0.5,
    max_fraction=0.05,
    prefix="",
):
    df = df.copy()

    if odds_over_col not in df.columns:
        df[odds_over_col] = -110
    if odds_under_col not in df.columns:
        df[odds_under_col] = -110

    df["IMPLIED_PROB_OVER"] = american_odds_to_implied_prob(df[odds_over_col])
    df["IMPLIED_PROB_UNDER"] = american_odds_to_implied_prob(df[odds_under_col])

    df["B_ODDS_OVER"] = american_odds_to_b(df[odds_over_col])
    df["B_ODDS_UNDER"] = american_odds_to_b(df[odds_under_col])

    edge_over_col = f"EDGE_OVER_{prefix}" if prefix else "EDGE_OVER"
    edge_under_col = f"EDGE_UNDER_{prefix}" if prefix else "EDGE_UNDER"
    bet_col = f"BET_PLACED_{prefix}" if prefix else "BET_PLACED"
    result_col = f"BET_RESULT_{prefix}" if prefix else "BET_RESULT"
    pnl_col = f"PNL_{prefix}" if prefix else "PNL"

    df[edge_over_col] = df[prob_over_col] - df["IMPLIED_PROB_OVER"]
    df[edge_under_col] = df[prob_under_col] - df["IMPLIED_PROB_UNDER"]

    df[bet_col] = "NO BET"
    df[bet_col] = np.where(df[edge_over_col] > edge_threshold, "OVER", df[bet_col])
    df[bet_col] = np.where(
        (df[edge_under_col] > edge_threshold) & (df[bet_col] == "NO BET"),
        "UNDER",
        df[bet_col],
    )

    df["KELLY_FRACTION"] = 0.0

    over_mask = df[bet_col] == "OVER"
    under_mask = df[bet_col] == "UNDER"

    df.loc[over_mask, "KELLY_FRACTION"] = (
        (df.loc[over_mask, "B_ODDS_OVER"] * df.loc[over_mask, prob_over_col]) -
        (1 - df.loc[over_mask, prob_over_col])
    ) / df.loc[over_mask, "B_ODDS_OVER"]

    df.loc[under_mask, "KELLY_FRACTION"] = (
        (df.loc[under_mask, "B_ODDS_UNDER"] * df.loc[under_mask, prob_under_col]) -
        (1 - df.loc[under_mask, prob_under_col])
    ) / df.loc[under_mask, "B_ODDS_UNDER"]

    df["KELLY_FRACTION"] = np.clip(df["KELLY_FRACTION"] * kelly_fraction_scale, 0, max_fraction)

    conditions = [
        (df[bet_col] == "OVER") & (df[actual_col] > df[line_col]),
        (df[bet_col] == "UNDER") & (df[actual_col] < df[line_col]),
        (df[bet_col] == "OVER") & (df[actual_col] < df[line_col]),
        (df[bet_col] == "UNDER") & (df[actual_col] > df[line_col]),
    ]
    df[result_col] = np.select(conditions, ["WIN", "WIN", "LOSS", "LOSS"], default="PUSH")

    df["WAGER_AMOUNT"] = np.where(df[bet_col] != "NO BET", starting_bankroll * df["KELLY_FRACTION"], 0)

    win_multiplier = np.where(df[bet_col] == "OVER", df["B_ODDS_OVER"], df["B_ODDS_UNDER"])
    pnl_conditions = [df[result_col] == "WIN", df[result_col] == "LOSS"]
    pnl_choices = [df["WAGER_AMOUNT"] * win_multiplier, -df["WAGER_AMOUNT"]]
    df[pnl_col] = np.select(pnl_conditions, pnl_choices, default=0)

    return df



def run_kelly_backtest(
    df,
    prob_over_col,
    prob_under_col,
    line_col="TRUE_MARKET_LINE",
    actual_col="PTS",
    odds_over_col="ODDS_OVER",
    odds_under_col="ODDS_UNDER",
    edge_threshold=0.03,
    starting_bankroll=10000,
    kelly_fraction_scale=0.5,
    max_fraction=0.05,
    prefix="",
):
    df = df.copy()

    if odds_over_col not in df.columns:
        df[odds_over_col] = -110
    if odds_under_col not in df.columns:
        df[odds_under_col] = -110

    df["IMPLIED_PROB_OVER"] = american_odds_to_implied_prob(df[odds_over_col])
    df["IMPLIED_PROB_UNDER"] = american_odds_to_implied_prob(df[odds_under_col])

    df["B_ODDS_OVER"] = american_odds_to_b(df[odds_over_col])
    df["B_ODDS_UNDER"] = american_odds_to_b(df[odds_under_col])

    edge_over_col = f"EDGE_OVER_{prefix}" if prefix else "EDGE_OVER"
    edge_under_col = f"EDGE_UNDER_{prefix}" if prefix else "EDGE_UNDER"
    bet_col = f"BET_PLACED_{prefix}" if prefix else "BET_PLACED"
    result_col = f"BET_RESULT_{prefix}" if prefix else "BET_RESULT"
    pnl_col = f"PNL_{prefix}" if prefix else "PNL"

    df[edge_over_col] = df[prob_over_col] - df["IMPLIED_PROB_OVER"]
    df[edge_under_col] = df[prob_under_col] - df["IMPLIED_PROB_UNDER"]

    df[bet_col] = "NO BET"
    df[bet_col] = np.where(df[edge_over_col] > edge_threshold, "OVER", df[bet_col])
    df[bet_col] = np.where(
        (df[edge_under_col] > edge_threshold) & (df[bet_col] == "NO BET"),
        "UNDER",
        df[bet_col],
    )

    df["KELLY_FRACTION"] = 0.0

    over_mask = df[bet_col] == "OVER"
    under_mask = df[bet_col] == "UNDER"

    df.loc[over_mask, "KELLY_FRACTION"] = (
        (df.loc[over_mask, "B_ODDS_OVER"] * df.loc[over_mask, prob_over_col]) -
        (1 - df.loc[over_mask, prob_over_col])
    ) / df.loc[over_mask, "B_ODDS_OVER"]

    df.loc[under_mask, "KELLY_FRACTION"] = (
        (df.loc[under_mask, "B_ODDS_UNDER"] * df.loc[under_mask, prob_under_col]) -
        (1 - df.loc[under_mask, prob_under_col])
    ) / df.loc[under_mask, "B_ODDS_UNDER"]

    df["KELLY_FRACTION"] = np.clip(df["KELLY_FRACTION"] * kelly_fraction_scale, 0, max_fraction)

    conditions = [
        (df[bet_col] == "OVER") & (df[actual_col] > df[line_col]),
        (df[bet_col] == "UNDER") & (df[actual_col] < df[line_col]),
        (df[bet_col] == "OVER") & (df[actual_col] < df[line_col]),
        (df[bet_col] == "UNDER") & (df[actual_col] > df[line_col]),
    ]
    df[result_col] = np.select(conditions, ["WIN", "WIN", "LOSS", "LOSS"], default="PUSH")

    df["WAGER_AMOUNT"] = np.where(df[bet_col] != "NO BET", starting_bankroll * df["KELLY_FRACTION"], 0)

    win_multiplier = np.where(df[bet_col] == "OVER", df["B_ODDS_OVER"], df["B_ODDS_UNDER"])
    pnl_conditions = [df[result_col] == "WIN", df[result_col] == "LOSS"]
    pnl_choices = [df["WAGER_AMOUNT"] * win_multiplier, -df["WAGER_AMOUNT"]]
    df[pnl_col] = np.select(pnl_conditions, pnl_choices, default=0)

    return df
