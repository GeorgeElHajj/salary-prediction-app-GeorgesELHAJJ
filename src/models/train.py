from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data


DATA_PATH = Path("data/raw/ds_salaries.csv")
MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "model.joblib"


def build_pipeline(categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def main():
    print("Loading dataset...")
    df = load_dataset(str(DATA_PATH))

    print("Preprocessing dataset...")
    X, y, processed_df = preprocess_data(df)

    # Save processed dataset (optional but useful)
    processed_output_path = Path("data/processed/cleaned_salaries.csv")
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_output_path, index=False)

    categorical_features = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "company_location",
        "company_size",
    ]

    numerical_features = [
        "work_year",
        "remote_ratio",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")

    pipeline = build_pipeline(categorical_features, numerical_features)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()