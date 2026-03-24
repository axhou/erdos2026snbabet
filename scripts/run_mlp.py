import pandas as pd

from src.config import PREDICTIONS_DIR, PROCESSED_DATA_DIR
from src.distributions import add_log_likelihood, add_probability_columns
from src.models.mlp_model import train_and_evaluate_mlp

if __name__ == "__main__":
    input_path = PROCESSED_DATA_DIR / "matchup_model_data.csv"
    df = pd.read_csv(input_path)

    mlp_df, model, scaler = train_and_evaluate_mlp(df, epochs=200)

    mlp_df = add_probability_columns(
        mlp_df,
        mu_col="MLP_ADJUSTED_MU",
        sigma2_col="MLP_ADJUSTED_SIGMA2",
        prefix="_MLP",
    )
    mlp_df = add_log_likelihood(
        mlp_df,
        actual_col="PTS",
        n_col="n_param_MLP",
        p_col="p_param_MLP",
        suffix="_MLP",
    )

    output_path = PREDICTIONS_DIR / "mlp_predictions.csv"
    mlp_df.to_csv(output_path, index=False)

    print(f"Saved MLP predictions to: {output_path}")
    print(f"Mean MLP NLL: {mlp_df['LOG_LIKELIHOOD_MLP'].mean():.4f}")
