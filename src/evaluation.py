from src.distributions import (
    add_log_likelihood,
    add_probability_columns,
    negative_binomial_params,
)
def evaluate_gam(df):
    df = df.copy()

    n_param, p_param = negative_binomial_params(
        df["GAM_ADJUSTED_MU"],
        df["GAM_ADJUSTED_SIGMA2"],
    )
    df["n_param_GAM"] = n_param
    df["p_param_GAM"] = p_param

    if "TRUE_MARKET_LINE" in df.columns:
        df = add_probability_columns(
            df,
            mu_col="GAM_ADJUSTED_MU",
            sigma2_col="GAM_ADJUSTED_SIGMA2",
            prefix="_GAM",
        )

    df = add_log_likelihood(
        df,
        actual_col="PTS",
        n_col="n_param_GAM",
        p_col="p_param_GAM",
        suffix="_GAM",
    )

    return df

def evaluate_glm(df):
    df = df.copy()

    n_param, p_param = negative_binomial_params(
        df["MATCHUP_ADJUSTED_MU"],
        df["MATCHUP_ADJUSTED_SIGMA2"],
    )
    df["n_param_GLM"] = n_param
    df["p_param_GLM"] = p_param

    if "TRUE_MARKET_LINE" in df.columns:
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


def evaluate_ensemble_nll(df):
    df = df.copy()

    n_param_rf, p_param_rf = negative_binomial_params(
        df["RF_ADJUSTED_MU"],
        df["RF_ADJUSTED_SIGMA2"],
    )
    df["n_param_RF"] = n_param_rf
    df["p_param_RF"] = p_param_rf

    if "TRUE_MARKET_LINE" in df.columns:
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

    n_param_xgb, p_param_xgb = negative_binomial_params(
        df["XGB_ADJUSTED_MU"],
        df["XGB_ADJUSTED_SIGMA2"],
    )
    df["n_param_XGB"] = n_param_xgb
    df["p_param_XGB"] = p_param_xgb

    if "TRUE_MARKET_LINE" in df.columns:
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


def evaluate_xgb_walk_forward(df):
    df = df.copy()

    n_param_xgb, p_param_xgb = negative_binomial_params(
        df["XGB_WF_ADJUSTED_MU"],
        df["XGB_WF_ADJUSTED_SIGMA2"],
    )
    df["n_param_XGB_WF"] = n_param_xgb
    df["p_param_XGB_WF"] = p_param_xgb

    if "TRUE_MARKET_LINE" in df.columns:
        df = add_probability_columns(
            df,
            mu_col="XGB_WF_ADJUSTED_MU",
            sigma2_col="XGB_WF_ADJUSTED_SIGMA2",
            prefix="_XGB_WF",
        )

    df = add_log_likelihood(
        df,
        actual_col="PTS",
        n_col="n_param_XGB_WF",
        p_col="p_param_XGB_WF",
        suffix="_XGB_WF",
    )

    return df