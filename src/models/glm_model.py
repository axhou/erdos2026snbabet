import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_glm_and_predict(train_df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """
    Fit Poisson GLM and create matchup-adjusted mean and variance columns.
    """
    train_df = train_df.copy()

    formula = (
        "PTS ~ OFF_RATING_ROLL_MEAN_10 + "
        "OPP_DEF_RATING_ROLL_MEAN_10 + "
        "PACE_ROLL_MEAN_10 + "
        "OPP_PACE_ROLL_MEAN_10"
    )

    print("Fitting Poisson GLM...")
    glm_model = smf.glm(
        formula=formula,
        data=train_df,
        family=sm.families.Poisson(link=sm.families.links.Log()),
    ).fit()

    train_df["MATCHUP_ADJUSTED_MU"] = glm_model.predict(train_df)

    dispersion_ratio = train_df["PTS_ROLL_VAR_SIGMA2"] / train_df["PTS_ROLL_MEAN_MU"]
    train_df["MATCHUP_ADJUSTED_SIGMA2"] = train_df["MATCHUP_ADJUSTED_MU"] * dispersion_ratio

    return train_df, glm_model
