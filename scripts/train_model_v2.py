import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.features import make_structured_features


DATA_PATH = "relevant_priors_public.json"
MODEL_PATH = "app/model_v2.joblib"


def load_training_rows():
    data = json.load(open(DATA_PATH, "r", encoding="utf-8"))

    truth = {
        (row["case_id"], row["study_id"]): int(row["is_relevant_to_current"])
        for row in data["truth"]
    }

    rows = []

    for case in data["cases"]:
        case_id = case["case_id"]
        current = case["current_study"]

        for prior in case["prior_studies"]:
            features = make_structured_features(
                current["study_description"],
                prior["study_description"],
                current.get("study_date", ""),
                prior.get("study_date", ""),
            )

            features["case_id"] = case_id
            features["study_id"] = prior["study_id"]
            features["label"] = truth[(case_id, prior["study_id"])]

            rows.append(features)

    return pd.DataFrame(rows)


def main():
    df = load_training_rows()

    numeric_features = [
        "same_modality",
        "token_jaccard",
        "same_body_region",
        "num_current_regions",
        "num_prior_regions",
        "days_between",
        "years_between",
        "heuristic_prediction",
    ]

    categorical_features = [
        "current_modality",
        "prior_modality",
        "current_regions",
        "prior_regions",
    ]

    text_feature = "pair_text"

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=60000,
                ),
                text_feature,
            ),
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    C=2.0,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, valid_idx = next(
        splitter.split(df, df["label"], groups=df["case_id"])
    )

    train_df = df.iloc[train_idx]
    valid_df = df.iloc[valid_idx]

    X_train = train_df.drop(columns=["label", "case_id", "study_id"])
    y_train = train_df["label"]

    X_valid = valid_df.drop(columns=["label", "case_id", "study_id"])
    y_valid = valid_df["label"]

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_valid)[:, 1]

    best_threshold = 0.5
    best_acc = 0.0

    for i in range(5, 96):
        threshold = i / 100
        preds = probs >= threshold
        acc = accuracy_score(y_valid, preds)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    final_preds = probs >= best_threshold

    print(f"Validation accuracy: {best_acc:.4f}")
    print(f"Best threshold: {best_threshold}")
    print(classification_report(y_valid, final_preds))

    artifact = {
        "model": model,
        "threshold": best_threshold,
        "feature_columns": list(X_train.columns),
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()