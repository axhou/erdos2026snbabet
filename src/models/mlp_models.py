import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_FEATURES


class NBAScoringMLP(nn.Module):
    def __init__(self, input_dim: int):
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


def train_and_evaluate_mlp(df: pd.DataFrame, epochs: int = 200):
    """
    Train a simple MLP to predict team points.
    """
    df = df.copy()
    df = df.dropna(subset=DEFAULT_FEATURES + ["PTS", "PTS_ROLL_VAR_SIGMA2", "PTS_ROLL_MEAN_MU"]).reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[DEFAULT_FEATURES])
    y = df["PTS"].values

    split_idx = int(len(df) * 0.8)
    X_train_tensor = torch.FloatTensor(X_scaled[:split_idx])
    y_train_tensor = torch.FloatTensor(y[:split_idx]).view(-1, 1)

    X_all_tensor = torch.FloatTensor(X_scaled)

    model = NBAScoringMLP(input_dim=len(DEFAULT_FEATURES))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training PyTorch MLP...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_all_tensor).numpy().flatten()

    df["MLP_ADJUSTED_MU"] = preds

    dispersion_ratio = df["PTS_ROLL_VAR_SIGMA2"] / df["PTS_ROLL_MEAN_MU"]
    df["MLP_ADJUSTED_SIGMA2"] = df["MLP_ADJUSTED_MU"] * dispersion_ratio

    return df, model, scaler
