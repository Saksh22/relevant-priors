import json
from pathlib import Path
import sys

# allow importing from app/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.schemas import Case
from app.model import predict_batch_v2


def main():
    data = json.load(open("relevant_priors_public.json", "r", encoding="utf-8"))

    truth = {
        (row["case_id"], row["study_id"]): row["is_relevant_to_current"]
        for row in data["truth"]
    }

    cases = [Case(**case) for case in data["cases"]]
    predictions = predict_batch_v2(cases)

    desc_lookup = {}
    for case in data["cases"]:
        current_desc = case["current_study"]["study_description"]

        for prior in case["prior_studies"]:
            desc_lookup[(case["case_id"], prior["study_id"])] = (
                current_desc,
                prior["study_description"],
            )

    correct = 0
    total = 0
    false_pos = []
    false_neg = []

    for prediction in predictions:
        key = (prediction["case_id"], prediction["study_id"])
        pred = prediction["predicted_is_relevant"]
        gold = truth[key]

        if pred == gold:
            correct += 1
        else:
            current_desc, prior_desc = desc_lookup[key]

            if pred:
                false_pos.append((current_desc, prior_desc))
            else:
                false_neg.append((current_desc, prior_desc))

        total += 1

    accuracy = correct / total

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"False Positives: {len(false_pos)}")
    print(f"False Negatives: {len(false_neg)}")

    print("\n--- Sample False Negatives (missed relevant) ---")
    for c, p in false_neg[:10]:
        print(f"{c}  <--  {p}")

    print("\n--- Sample False Positives (wrongly marked relevant) ---")
    for c, p in false_pos[:10]:
        print(f"{c}  <--  {p}")


if __name__ == "__main__":
    main()