import re
from pathlib import Path
import joblib

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


def predict_one_ml(current_desc: str, prior_desc: str) -> bool:
    artifact = get_trained_artifact()

    heuristic_pred = predict_one(current_desc, prior_desc)

    if artifact is None:
        return heuristic_pred

    text = f"CURRENT: {current_desc} PRIOR: {prior_desc}"
    prob = artifact["model"].predict_proba([text])[0, 1]

    if prob >= 0.90:
        return True

    if prob <= 0.05:
        return False

    return heuristic_pred