import re
from pathlib import Path
import joblib
from app.features import make_structured_features

MODEL_V2_PATH = Path(__file__).resolve().parent / "model_v2.joblib"

_model_v2_artifact = None

TRAINED_MODEL_PATH = Path(__file__).resolve().parent / "trained_model.joblib"

_trained_artifact = None

STOPWORDS = {
    "with", "without", "w", "wo", "cntrst", "contrast", "con",
    "and", "or", "limited", "complete", "exam", "study", "portable",
    "procedure"
}

MODALITY_ALIASES = {
    "mri": "mr",
    "mr": "mr",
    "ct": "ct",
    "xr": "xr",
    "xray": "xr",
    "radiograph": "xr",
    "us": "us",
    "ultrasound": "us",
    "nm": "nm",
    "pet": "pet",
    "mam": "mammo",
    "mammo": "mammo",
    "mammogram": "mammo",
    "dxa": "dxa",
    "echo": "echo",
}


BODY_GROUPS = [
    {"brain", "head", "stroke", "skull"},
    {"neck", "cervical", "cspine"},
    {"chest", "thorax", "lung", "ribs"},
    {"abdomen", "abdominal"},
    {"pelvis", "pelvic"},
    {"spine", "lumbar", "thoracic"},
    {"knee"},
    {"shoulder"},
    {"hip"},
    {"ankle", "foot"},
    {"hand", "wrist"},
    {"breast", "mammo", "mammogram"},
]

BREAST_TERMS = {
    "mam",
    "mammo",
    "mammography",
    "mammogram",
    "breast",
    "tomo",
}

BREAST_ULTRASOUND_TERMS = {
    "us breast",
    "ultrasound breast",
    "breast ultrasound",
}

BREAST_MRI_TERMS = {
    "mri breast",
    "mr breast",
}

CARDIAC_TERMS = {
    "myo",
    "myocardial",
    "coronary",
    "cardiac",
    "calcium",
    "calc",
    "cta coronary",
    "ct coronary",
}

PET_CT_TERMS = {
    "pet ct",
    "pet/ct",
    "skullthigh",
    "skull thigh",
    "f18",
}

def normalize(text: str) -> list[str]:
    text = text or ""
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [t for t in text.split() if t and t not in STOPWORDS]


def modality(desc: str) -> str:
    tokens = normalize(desc)

    for token in tokens[:4]:
        if token in MODALITY_ALIASES:
            return MODALITY_ALIASES[token]

    for token in tokens:
        if token in MODALITY_ALIASES:
            return MODALITY_ALIASES[token]

    return ""


def body_terms(desc: str) -> set[str]:
    tokens = set(normalize(desc))
    return tokens - set(MODALITY_ALIASES.keys())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)


def shares_body_group(current_terms: set[str], prior_terms: set[str]) -> bool:
    for group in BODY_GROUPS:
        if current_terms & group and prior_terms & group:
            return True

    return False


def predict_one(current_desc: str, prior_desc: str) -> bool:
    current_modality = modality(current_desc)
    prior_modality = modality(prior_desc)

    current_terms = body_terms(current_desc)
    prior_terms = body_terms(prior_desc)

    similarity = jaccard(current_terms, prior_terms)

    if current_modality and prior_modality and current_modality == prior_modality and similarity >= 0.35:
        return True

    if shares_body_group(current_terms, prior_terms):
        return True

    return False

def get_trained_artifact():
    global _trained_artifact

    if _trained_artifact is None and TRAINED_MODEL_PATH.exists():
        _trained_artifact = joblib.load(TRAINED_MODEL_PATH)

    return _trained_artifact


def contains_any(desc: str, terms: set[str]) -> bool:
    normalized = " ".join(normalize(desc))
    return any(term in normalized for term in terms)

def is_breast_related(desc: str) -> bool:
    return (
        contains_any(desc, BREAST_TERMS)
        or contains_any(desc, BREAST_ULTRASOUND_TERMS)
        or contains_any(desc, BREAST_MRI_TERMS)
    )


def is_cardiac_related(desc: str) -> bool:
    return contains_any(desc, CARDIAC_TERMS)


def is_petct_related(desc: str) -> bool:
    return contains_any(desc, PET_CT_TERMS)


def targeted_override(current_desc: str, prior_desc: str) -> bool | None:
    """
    Return:
      True  -> force relevant
      False -> force irrelevant
      None  -> no override
    """

    # Breast priors are often relevant across mammo / US breast / MRI breast.
    if is_breast_related(current_desc) and is_breast_related(prior_desc):
        return True

    # Myocardial perfusion and coronary CT/calcium scoring are related priors.
    if is_cardiac_related(current_desc) and is_cardiac_related(prior_desc):
        return True

    # PET/CT is often relevant to CT chest/abdomen/pelvis oncology-style follow-up.
    if is_petct_related(current_desc) or is_petct_related(prior_desc):
        current_terms = body_terms(current_desc)
        prior_terms = body_terms(prior_desc)

        body_overlap = {
            "chest",
            "lung",
            "thorax",
            "abdomen",
            "abdominal",
            "pelvis",
            "pelvic",
        }

        if current_terms & body_overlap or prior_terms & body_overlap:
            return True

    return None

def get_model_v2_artifact():
    global _model_v2_artifact

    if _model_v2_artifact is None and MODEL_V2_PATH.exists():
        _model_v2_artifact = joblib.load(MODEL_V2_PATH)

    return _model_v2_artifact


def predict_batch_v2(cases) -> list[dict]:
    artifact = get_model_v2_artifact()

    rows = []
    prediction_refs = []

    for case in cases:
        for prior in case.prior_studies:
            rows.append(
                make_structured_features(
                    case.current_study.study_description,
                    prior.study_description,
                    case.current_study.study_date,
                    prior.study_date,
                )
            )

            prediction_refs.append(
                {
                    "case_id": case.case_id,
                    "study_id": prior.study_id,
                }
            )

    if not rows:
        return []

    if artifact is None:
        results = []

        for case in cases:
            for prior in case.prior_studies:
                results.append(
                    {
                        "case_id": case.case_id,
                        "study_id": prior.study_id,
                        "predicted_is_relevant": predict_one_ml(
                            case.current_study.study_description,
                            prior.study_description,
                        ),
                    }
                )

        return results

    import pandas as pd

    X = pd.DataFrame(rows)
    probs = artifact["model"].predict_proba(X)[:, 1]
    threshold = artifact["threshold"]

    predictions = []

    for ref, prob in zip(prediction_refs, probs):
        predictions.append(
            {
                "case_id": ref["case_id"],
                "study_id": ref["study_id"],
                "predicted_is_relevant": bool(prob >= threshold),
            }
        )

    return predictions