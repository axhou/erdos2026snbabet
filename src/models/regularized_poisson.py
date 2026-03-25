import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_FEATURES


def train_regularized_model(df: pd.DataFrame):
    """
    Train regularized Poisson regression.
    """
    df = df.copy().dropna(subset=DEFAULT_FEATURES + ["PTS"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[DEFAULT_FEATURES])
    y = df["PTS"].values

    split_idx = int(len(df) * 0.8)
    X_train, y_train = X_scaled[:split_idx], y[:split_idx]

    print("Training Regularized Poisson Model...")
    reg_model = PoissonRegressor(alpha=0.5, max_iter=1000)
    reg_model.fit(X_train, y_train)

    df["REGULARIZED_MU"] = reg_model.predict(X_scaled)
    return df, reg_model, scaler
