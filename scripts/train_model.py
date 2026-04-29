import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.model import predict_one


DATA_PATH = "relevant_priors_public.json"
MODEL_PATH = "app/trained_model.joblib"


def make_pair_text(current_desc: str, prior_desc: str) -> str:
    return f"CURRENT: {current_desc} PRIOR: {prior_desc}"


def load_rows():
    data = json.load(open(DATA_PATH, "r", encoding="utf-8"))

    truth = {
        (row["case_id"], row["study_id"]): row["is_relevant_to_current"]
        for row in data["truth"]
    }

    rows = []

    for case in data["cases"]:
        case_id = case["case_id"]
        current_desc = case["current_study"]["study_description"]

        for prior in case["prior_studies"]:
            prior_desc = prior["study_description"]
            study_id = prior["study_id"]

            rows.append(
                {
                    "case_id": case_id,
                    "study_id": study_id,
                    "text": make_pair_text(current_desc, prior_desc),
                    "heuristic": int(predict_one(current_desc, prior_desc)),
                    "label": int(truth[(case_id, study_id)]),
                }
            )

    return pd.DataFrame(rows)


def main():
    df = load_rows()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(df, df["label"], groups=df["case_id"])
    )

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=2.0,
                ),
            ),
        ]
    )

    model.fit(train_df["text"], train_df["label"])

    probs = model.predict_proba(test_df["text"])[:, 1]

    best_acc = 0
    best_threshold = 0.5

    for threshold in [i / 100 for i in range(5, 96)]:
        preds = probs >= threshold
        acc = accuracy_score(test_df["label"], preds)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    preds = probs >= best_threshold

    print(f"Validation accuracy: {best_acc:.4f}")
    print(f"Best threshold: {best_threshold}")
    print(classification_report(test_df["label"], preds))

    artifact = {
        "model": model,
        "threshold": best_threshold,
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()