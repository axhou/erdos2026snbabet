import pandas as pd

from src.config import PREDICTIONS_DIR, PROCESSED_DATA_DIR
from src.evaluation import evaluate_glm
from src.models.glm_model import fit_glm_and_predict

if __name__ == "__main__":
    input_path = PROCESSED_DATA_DIR / "matchup_model_data.csv"
    df = pd.read_csv(input_path)

    glm_df, glm_model = fit_glm_and_predict(df)
    glm_df = evaluate_glm(glm_df)

    output_path = PREDICTIONS_DIR / "glm_predictions.csv"
    glm_df.to_csv(output_path, index=False)

    print(f"Saved GLM predictions to: {output_path}")
    print(glm_model.summary())
