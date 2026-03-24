# src/models.py

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


GLM_FORMULA = "PTS ~ OFF_RATING_ROLL_MEAN_10 + OPP_DEF_RATING_ROLL_MEAN_10 + PACE_ROLL_MEAN_10 + OPP_PACE_ROLL_MEAN_10"


def fit_glm_and_predict(train_df, test_df, formula=GLM_FORMULA):
    train_df = train_df.copy()
    test_df = test_df.copy()

    glm_model = smf.glm(
        formula=formula,
        data=train_df,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit()

    train_df["MATCHUP_ADJUSTED_MU"] = glm_model.predict(train_df)
    test_df["MATCHUP_ADJUSTED_MU"] = glm_model.predict(test_df)

    for df_ in [train_df, test_df]:
        dispersion_ratio = df_["PTS_ROLL_VAR_SIGMA2"] / df_["PTS_ROLL_MEAN_MU"]
        df_["MATCHUP_ADJUSTED_SIGMA2"] = df_["MATCHUP_ADJUSTED_MU"] * dispersion_ratio

    return train_df, test_df, glm_model


from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "OFF_RATING_ROLL_MEAN_10",
    "OPP_DEF_RATING_ROLL_MEAN_10",
    "PACE_ROLL_MEAN_10",
    "OPP_PACE_ROLL_MEAN_10",
]


def fit_regularized_poisson(train_df, test_df, features=DEFAULT_FEATURES, alpha=0.5):
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    y_train = train_df["PTS"].values

    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(X_train, y_train)

    train_df["REGULARIZED_MU"] = model.predict(X_train)
    test_df["REGULARIZED_MU"] = model.predict(X_test)

    return train_df, test_df, model, scaler

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def fit_random_forest(train_df, test_df, features=DEFAULT_FEATURES):
    train_df = train_df.copy()
    test_df = test_df.copy()

    X_train = train_df[features]
    y_train = train_df["PTS"]
    X_test = test_df[features]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_df["RF_ADJUSTED_MU"] = model.predict(X_train)
    test_df["RF_ADJUSTED_MU"] = model.predict(X_test)

    for df_ in [train_df, test_df]:
        dispersion_ratio = df_["PTS_ROLL_VAR_SIGMA2"] / df_["PTS_ROLL_MEAN_MU"]
        df_["RF_ADJUSTED_SIGMA2"] = df_["RF_ADJUSTED_MU"] * dispersion_ratio

    return train_df, test_df, model


def fit_xgboost(train_df, test_df, features=DEFAULT_FEATURES):
    train_df = train_df.copy()
    test_df = test_df.copy()

    X_train = train_df[features]
    y_train = train_df["PTS"]
    X_test = test_df[features]

    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_df["XGB_ADJUSTED_MU"] = model.predict(X_train)
    test_df["XGB_ADJUSTED_MU"] = model.predict(X_test)

    for df_ in [train_df, test_df]:
        dispersion_ratio = df_["PTS_ROLL_VAR_SIGMA2"] / df_["PTS_ROLL_MEAN_MU"]
        df_["XGB_ADJUSTED_SIGMA2"] = df_["XGB_ADJUSTED_MU"] * dispersion_ratio

    return train_df, test_df, model

import torch
import torch.nn as nn


class NBAScoringMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(8, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.softplus(self.output_layer(x))
        return x

import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


def fit_mlp(train_df, test_df, features=DEFAULT_FEATURES, epochs=150, lr=0.01):
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    y_train = train_df["PTS"].values

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)

    model = NBAScoringMLP(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_df["NN_ADJUSTED_MU"] = model(X_train_tensor).numpy().flatten()
        test_df["NN_ADJUSTED_MU"] = model(X_test_tensor).numpy().flatten()

    for df_ in [train_df, test_df]:
        dispersion_ratio = df_["PTS_ROLL_VAR_SIGMA2"] / df_["PTS_ROLL_MEAN_MU"]
        df_["NN_ADJUSTED_SIGMA2"] = df_["NN_ADJUSTED_MU"] * dispersion_ratio

    return train_df, test_df, model, scaler

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def fit_random_forest(train_df, test_df, features=DEFAULT_FEATURES):
    train_df = train_df.copy()
    test_df = test_df.copy()

    X_train = train_df[features]
    y_train = train_df["PTS"]
    X_test = test_df[features]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_df["RF_ADJUSTED_MU"] = model.predict(X_train)
    test_df["RF_ADJUSTED_MU"] = model.predict(X_test)

    for df_ in [train_df, test_df]:
        dispersion_ratio = df_["PTS_ROLL_VAR_SIGMA2"] / df_["PTS_ROLL_MEAN_MU"]
        df_["RF_ADJUSTED_SIGMA2"] = df_["RF_ADJUSTED_MU"] * dispersion_ratio

    return train_df, test_df, model


def fit_xgboost(train_df, test_df, features=DEFAULT_FEATURES):
    train_df = train_df.copy()
    test_df = test_df.copy()

    X_train = train_df[features]
    y_train = train_df["PTS"]
    X_test = test_df[features]

    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_df["XGB_ADJUSTED_MU"] = model.predict(X_train)
    test_df["XGB_ADJUSTED_MU"] = model.predict(X_test)

    for df_ in [train_df, test_df]:
        dispersion_ratio = df_["PTS_ROLL_VAR_SIGMA2"] / df_["PTS_ROLL_MEAN_MU"]
        df_["XGB_ADJUSTED_SIGMA2"] = df_["XGB_ADJUSTED_MU"] * dispersion_ratio

    return train_df, test_df, model

import torch
import torch.nn as nn


class NBAScoringMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(8, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.softplus(self.output_layer(x))
        return x

import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


def fit_mlp(train_df, test_df, features=DEFAULT_FEATURES, epochs=150, lr=0.01):
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    y_train = train_df["PTS"].values

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)

    model = NBAScoringMLP(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_df["NN_ADJUSTED_MU"] = model(X_train_tensor).numpy().flatten()
        test_df["NN_ADJUSTED_MU"] = model(X_test_tensor).numpy().flatten()

    for df_ in [train_df, test_df]:
        dispersion_ratio = df_["PTS_ROLL_VAR_SIGMA2"] / df_["PTS_ROLL_MEAN_MU"]
        df_["NN_ADJUSTED_SIGMA2"] = df_["NN_ADJUSTED_MU"] * dispersion_ratio

    return train_df, test_df, model, scaler
