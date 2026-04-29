import json
from pathlib import Path
import sys

# allow importing from app/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.model import predict_one


def main():
    data = json.load(open("relevant_priors_public.json"))

    # Build truth lookup
    truth = {
        (row["case_id"], row["study_id"]): row["is_relevant_to_current"]
        for row in data["truth"]
    }

    correct = 0
    total = 0

    false_pos = []
    false_neg = []

    for case in data["cases"]:
        current_desc = case["current_study"]["study_description"]

        for prior in case["prior_studies"]:
            key = (case["case_id"], prior["study_id"])

            pred = predict_one(current_desc, prior["study_description"])
            gold = truth[key]

            if pred == gold:
                correct += 1
            else:
                if pred:
                    false_pos.append((current_desc, prior["study_description"]))
                else:
                    false_neg.append((current_desc, prior["study_description"]))

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