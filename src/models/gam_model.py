import pandas as pd
import numpy as np

from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.genmod.families import Poisson

from src.config import DEFAULT_FEATURES


def fit_gam_and_predict(df: pd.DataFrame):
    """
    Fit a Poisson GAM using spline bases on the default continuous features.
    """
    df = df.copy().dropna(
        subset=DEFAULT_FEATURES + ["PTS", "PTS_ROLL_VAR_SIGMA2", "PTS_ROLL_MEAN_MU"]
    ).reset_index(drop=True)

    X = df[DEFAULT_FEATURES].astype(float)
    y = df["PTS"].astype(float).values

    x_spline = X.values

    # one spline basis per continuous feature
    bs = BSplines(
        x_spline,
        df=[6] * len(DEFAULT_FEATURES),
        degree=[3] * len(DEFAULT_FEATURES),
    )

    print("Fitting Poisson GAM...")
    gam_model = GLMGam(
        y,
        exog=np.ones((len(df), 1)),   # intercept only in linear part
        smoother=bs,
        family=Poisson(),
    ).fit()

    df["GAM_ADJUSTED_MU"] = gam_model.predict(
        exog=np.ones((len(df), 1)),
        exog_smooth=x_spline
    )

    dispersion_ratio = df["PTS_ROLL_VAR_SIGMA2"] / df["PTS_ROLL_MEAN_MU"]
    df["GAM_ADJUSTED_SIGMA2"] = df["GAM_ADJUSTED_MU"] * dispersion_ratio

    return df, gam_model