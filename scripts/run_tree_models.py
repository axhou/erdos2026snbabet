import pandas as pd

from src.config import PREDICTIONS_DIR, PROCESSED_DATA_DIR
from src.evaluation import evaluate_ensemble_nll
from src.models.tree_models import train_ensemble_models

if __name__ == "__main__":
    input_path = PROCESSED_DATA_DIR / "matchup_model_data.csv"
    df = pd.read_csv(input_path)

    ensemble_df, rf_model, xgb_model = train_ensemble_models(df)
    ensemble_df = evaluate_ensemble_nll(ensemble_df)

    output_path = PREDICTIONS_DIR / "tree_model_predictions.csv"
    ensemble_df.to_csv(output_path, index=False)

    print(f"Saved tree model predictions to: {output_path}")
    print(f"Mean RF NLL: {ensemble_df['LOG_LIKELIHOOD_RF'].mean():.4f}")
    print(f"Mean XGB NLL: {ensemble_df['LOG_LIKELIHOOD_XGB'].mean():.4f}")
