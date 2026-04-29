import re
from datetime import datetime


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
    "mammography": "mammo",
    "mammogram": "mammo",
    "echo": "echo",
}

BODY_GROUPS = {
    "brain_head": {"brain", "head", "stroke", "skull"},
    "neck": {"neck", "cervical", "carotid"},
    "chest": {"chest", "thorax", "lung", "ribs"},
    "abdomen": {"abdomen", "abdominal"},
    "pelvis": {"pelvis", "pelvic"},
    "spine": {"spine", "lumbar", "thoracic", "cervical"},
    "breast": {"breast", "mammo", "mammography", "mammogram", "tomo"},
    "cardiac": {"cardiac", "heart", "coronary", "myocardial", "myo"},
    "renal": {"renal", "kidney", "kidneys", "bladder"},
    "hip": {"hip"},
    "knee": {"knee"},
    "shoulder": {"shoulder"},
    "hand_wrist": {"hand", "wrist", "finger"},
    "foot_ankle": {"foot", "ankle"},
}


def normalize(text: str) -> list[str]:
    text = text or ""
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [t for t in text.split() if t and t not in STOPWORDS]


def joined(text: str) -> str:
    return " ".join(normalize(text))


def modality(desc: str) -> str:
    toks = normalize(desc)
    text = " ".join(toks)

    if "pet" in text and "ct" in text:
        return "petct"

    for tok in toks:
        if tok in MODALITY_ALIASES:
            return MODALITY_ALIASES[tok]

    return "unknown"


def body_regions(desc: str) -> set[str]:
    toks = set(normalize(desc))
    regions = set()

    for region, terms in BODY_GROUPS.items():
        if toks & terms:
            regions.add(region)

    return regions


def token_jaccard(a: str, b: str) -> float:
    ta = set(normalize(a))
    tb = set(normalize(b))

    if not ta and not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def safe_days_between(current_date: str, prior_date: str) -> int:
    try:
        c = datetime.fromisoformat(current_date).date()
        p = datetime.fromisoformat(prior_date).date()
        return abs((c - p).days)
    except Exception:
        return -1


def simple_heuristic(current_desc: str, prior_desc: str) -> int:
    cur_mod = modality(current_desc)
    pri_mod = modality(prior_desc)

    cur_regions = body_regions(current_desc)
    pri_regions = body_regions(prior_desc)

    jac = token_jaccard(current_desc, prior_desc)

    if cur_mod == pri_mod and jac >= 0.35:
        return 1

    if cur_regions & pri_regions:
        return 1

    return 0


def make_pair_text(current_desc: str, prior_desc: str) -> str:
    return f"CURRENT {current_desc} PRIOR {prior_desc}"


def make_structured_features(
    current_desc: str,
    prior_desc: str,
    current_date: str,
    prior_date: str,
) -> dict:
    cur_mod = modality(current_desc)
    pri_mod = modality(prior_desc)

    cur_regions = body_regions(current_desc)
    pri_regions = body_regions(prior_desc)

    days = safe_days_between(current_date, prior_date)

    return {
        "current_modality": cur_mod,
        "prior_modality": pri_mod,
        "same_modality": int(cur_mod == pri_mod),
        "token_jaccard": token_jaccard(current_desc, prior_desc),
        "same_body_region": int(bool(cur_regions & pri_regions)),
        "num_current_regions": len(cur_regions),
        "num_prior_regions": len(pri_regions),
        "days_between": days,
        "years_between": days / 365.25 if days >= 0 else -1,
        "heuristic_prediction": simple_heuristic(current_desc, prior_desc),
        "current_regions": "|".join(sorted(cur_regions)) or "none",
        "prior_regions": "|".join(sorted(pri_regions)) or "none",
        "pair_text": make_pair_text(current_desc, prior_desc),
    }