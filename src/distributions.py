import numpy as np
import pandas as pd
import scipy.stats as stats


def ensure_valid_variance(mu: pd.Series, sigma2: pd.Series) -> pd.Series:
    """
    Negative binomial requires variance > mean.
    """
    return np.where(sigma2 <= mu, mu + 0.01, sigma2)


def negative_binomial_params(mu: pd.Series, sigma2: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert mean/variance parameterization into (n, p) for scipy.stats.nbinom.
    """
    sigma2 = ensure_valid_variance(mu, sigma2)
    n_param = (mu ** 2) / (sigma2 - mu)
    p_param = mu / sigma2
    return n_param, p_param


def add_probability_columns(
    df: pd.DataFrame,
    mu_col: str,
    sigma2_col: str,
    line_col: str = "TRUE_MARKET_LINE",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Add NB probabilities and quartiles using model mean/variance columns.
    """
    df = df.copy()

    n_param, p_param = negative_binomial_params(df[mu_col], df[sigma2_col])
    floor_line = np.floor(df[line_col])

    df[f"n_param{prefix}"] = n_param
    df[f"p_param{prefix}"] = p_param
    df[f"PROB_OVER{prefix}"] = 1 - stats.nbinom.cdf(k=floor_line, n=n_param, p=p_param)
    df[f"PROB_UNDER{prefix}"] = stats.nbinom.cdf(k=floor_line, n=n_param, p=p_param)
    df[f"PROJECTED_SCORE_25TH{prefix}"] = stats.nbinom.ppf(0.25, n=n_param, p=p_param)
    df[f"PROJECTED_SCORE_75TH{prefix}"] = stats.nbinom.ppf(0.75, n=n_param, p=p_param)

    return df


def add_log_likelihood(
    df: pd.DataFrame,
    actual_col: str,
    n_col: str,
    p_col: str,
    suffix: str = "",
) -> pd.DataFrame:
    df = df.copy()
    actual_prob = stats.nbinom.pmf(k=df[actual_col], n=df[n_col], p=df[p_col])
    df[f"ACTUAL_SCORE_PROB{suffix}"] = actual_prob
    df[f"LOG_LIKELIHOOD{suffix}"] = -np.log(actual_prob + 1e-9)
    return df
