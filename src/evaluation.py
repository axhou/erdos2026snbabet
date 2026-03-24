import pandas as pd

from src.distributions import add_log_likelihood, add_probability_columns


def evaluate_glm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_probability_columns(
        df,
        mu_col="MATCHUP_ADJUSTED_MU",
        sigma2_col="MATCHUP_ADJUSTED_SIGMA2",
        prefix="_GLM",
    )
    df = add_log_likelihood(
        df,
        actual_col="PTS",
        n_col="n_param_GLM",
        p_col="p_param_GLM",
        suffix="_GLM",
    )
    return df


def evaluate_ensemble_nll(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_probability_columns(
        df,
        mu_col="RF_ADJUSTED_MU",
        sigma2_col="RF_ADJUSTED_SIGMA2",
        prefix="_RF",
    )
    df = add_log_likelihood(
        df,
        actual_col="PTS",
        n_col="n_param_RF",
        p_col="p_param_RF",
        suffix="_RF",
    )

    df = add_probability_columns(
        df,
        mu_col="XGB_ADJUSTED_MU",
        sigma2_col="XGB_ADJUSTED_SIGMA2",
        prefix="_XGB",
    )
    df = add_log_likelihood(
        df,
        actual_col="PTS",
        n_col="n_param_XGB",
        p_col="p_param_XGB",
        suffix="_XGB",
    )

    return df
