# src/probability.py

import numpy as np
from scipy import stats


def ensure_valid_variance(df, mean_col, var_col, eps=0.01):
    df = df.copy()
    df[var_col] = np.where(df[var_col] <= df[mean_col], df[mean_col] + eps, df[var_col])
    return df


def add_negative_binomial_params(df, mean_col, var_col, prefix):
    df = ensure_valid_variance(df, mean_col, var_col)

    df[f"n_param_{prefix}"] = (df[mean_col] ** 2) / (df[var_col] - df[mean_col])
    df[f"p_param_{prefix}"] = df[mean_col] / df[var_col]
    return df


def add_over_under_probabilities(df, n_col, p_col, line_col="TRUE_MARKET_LINE", prefix=""):
    df = df.copy()
    floor_line = np.floor(df[line_col])

    over_col = f"PROB_OVER_{prefix}" if prefix else "PROB_OVER"
    under_col = f"PROB_UNDER_{prefix}" if prefix else "PROB_UNDER"
    q25_col = f"PROJECTED_SCORE_25TH_{prefix}" if prefix else "PROJECTED_SCORE_25TH"
    q75_col = f"PROJECTED_SCORE_75TH_{prefix}" if prefix else "PROJECTED_SCORE_75TH"

    df[over_col] = 1 - stats.nbinom.cdf(k=floor_line, n=df[n_col], p=df[p_col])
    df[under_col] = stats.nbinom.cdf(k=floor_line, n=df[n_col], p=df[p_col])
    df[q25_col] = stats.nbinom.ppf(0.25, n=df[n_col], p=df[p_col])
    df[q75_col] = stats.nbinom.ppf(0.75, n=df[n_col], p=df[p_col])

    return df


def add_log_likelihood(df, actual_col, n_col, p_col, prefix=""):
    df = df.copy()
    prob_col = f"ACTUAL_SCORE_PROB_{prefix}" if prefix else "ACTUAL_SCORE_PROB"
    ll_col = f"LOG_LIKELIHOOD_{prefix}" if prefix else "LOG_LIKELIHOOD"

    df[prob_col] = stats.nbinom.pmf(k=df[actual_col], n=df[n_col], p=df[p_col])
    df[ll_col] = -np.log(df[prob_col] + 1e-9)
    return df
