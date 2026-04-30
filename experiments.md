# Experiments

## Approach

The task is to determine whether a prior study is relevant to a current study for the same patient. I framed this as a **binary classification problem on study pairs**, where each (current study, prior study) pair is an independent sample.

### Key design decisions:

* Treat each pair independently rather than ranking all priors jointly
* Combine **textual similarity + structured clinical features**
* Use **batch inference** for performance and scalability
* Validate using **Group-based splitting by case_id** to avoid leakage

### Final pipeline:

```text
Input JSON
→ Extract (current, prior) pairs
→ Generate features:
   - text features (TF-IDF on combined descriptions)
   - structured features (modality, body region, time delta, etc.)
→ Logistic Regression model
→ Probability thresholding
→ Return predictions for each prior
```

---

## Dataset

* 996 cases
* 27,614 prior studies
* Highly imbalanced (majority are not relevant)

---

## Baseline

**Majority Class (all False)**

| Metric   | Value                            |
| -------- | -------------------------------- |
| Accuracy | ~0.76                            |
| Behavior | Predict all priors as irrelevant |

This baseline performs surprisingly well due to class imbalance but has **zero recall**.

---

## Heuristic Model

Rules based on:

* modality matching
* token overlap (Jaccard similarity)
* body-region grouping

### Performance

| Metric          | Value      |
| --------------- | ---------- |
| Accuracy        | **0.8716** |
| False Positives | 573        |
| False Negatives | 2972       |

### Observations

* Strong improvement over baseline
* Good recall for same-modality studies
* Poor handling of:

  * cross-modality relevance (PET/CT vs CT)
  * inconsistent naming (mammography variants)

---

## Hybrid Model (Heuristic + ML Override)

Added a TF-IDF + Logistic Regression model:

* ML used only for **high-confidence overrides**
* fallback to heuristic otherwise

### Performance

| Metric          | Value      |
| --------------- | ---------- |
| Accuracy        | **0.8999** |
| False Positives | 604        |
| False Negatives | 2159       |

### Observations

* Improved recall significantly
* ML captured text patterns missed by rules
* Still dependent on heuristic structure

---

## Targeted Domain Rules

Added a small number of **high-impact domain rules**:

* Breast imaging (mammography, ultrasound, MRI breast)
* Cardiac imaging (myocardial perfusion, coronary CT)
* PET/CT ↔ CT for oncology

### Performance

| Metric          | Value      |
| --------------- | ---------- |
| Accuracy        | **0.9270** |
| False Positives | 752        |
| False Negatives | 1264       |

### Observations

* Large gain in recall
* Some increase in false positives
* Not scalable (manual rules)

---

## Final Model — Feature-based ML (v2)

Replaced rule-heavy logic with a structured ML pipeline.

### Features

**Text features**

* TF-IDF on combined current/prior descriptions (1–3 grams)

**Structured features**

* modality (current & prior)
* modality match
* body-region extraction
* body-region overlap
* token similarity (Jaccard)
* number of regions per study
* time difference (days, years)
* heuristic prediction (as feature)
* categorical encoding of region/modality combinations

### Model

* Logistic Regression
* class_weight="balanced"
* threshold tuned via validation

### Validation strategy

* Grouped split by `case_id`
* Prevents leakage across priors of same case

---

## Final Performance (Public Evaluation)

| Metric          | Value      |
| --------------- | ---------- |
| Accuracy        | **0.9373** |
| Correct         | 25,882     |
| Total           | 27,614     |
| False Positives | 594        |
| False Negatives | 1,138      |

### Improvements over baseline

| Stage             | Accuracy   |
| ----------------- | ---------- |
| Majority baseline | ~0.76      |
| Heuristic         | 0.8716     |
| Hybrid            | 0.8999     |
| Domain rules      | 0.9270     |
| Final ML (v2)     | **0.9373** |

---

## What Worked

* Treating the problem as **pairwise classification**
* Combining **text + structured features**
* Using **heuristic output as a feature**, not final logic
* Group-based validation to avoid leakage
* Batch inference (critical for performance constraints)

---

## What Failed

* Pure rule-based systems (not scalable, brittle)
* ML-only model without structure (~0.85 accuracy)
* Character n-grams (reduced recall, worse accuracy)
* Overly strict filtering (caused large false negative spikes)

---

## Error Analysis

Remaining errors are mostly:

### False Negatives

* Breast imaging variants with inconsistent naming
* Cross-modality relationships (echo vs CT chest)
* Generic descriptions (e.g., “BREAST”)

### False Positives

* Same modality but different anatomy
* Laterality conflicts (left vs right)
* Broad anatomical matches (abdomen vs pelvis vs MRI pelvis)

---

## Performance & Scalability

* Uses **batch inference** (single model call per request)
* Avoids per-prior model calls (prevents timeout risk)
* Fully local model (no external dependencies)
* Latency scales linearly with number of priors

---

## How I Would Improve Further

1. Replace Logistic Regression with **LightGBM** for nonlinear feature interactions
2. Normalize medical abbreviations more aggressively
3. Add **laterality-aware constraints** (left/right mismatch penalties)
4. Move from classification → **ranking model** (score priors within a case)
5. Use contextual embeddings (e.g., clinical BERT) instead of TF-IDF
6. Calibrate probabilities using cross-validation

---

## Summary

The final system is:

* Accurate (~0.94)
* Fast (batch inference)
* Generalizable (feature-based ML, not rules)

The approach prioritizes robustness and scalability over manual rule tuning, making it suitable for real-world deployment.
