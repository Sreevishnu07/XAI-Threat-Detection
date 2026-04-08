THREAT_LABELS = {
    "revolver": 1.0,
    "pistol": 1.0,
    "rifle": 1.0,
    "gun": 1.0,

    "knife": 0.9,
    "cleaver": 0.9,
    "dagger": 0.9,

    "scissors": 0.6,

    "dog": 0.1,
    "cat": 0.05,
    "person": 0.2,
}

def get_label_threat_weight(label: str) -> float:
    label = label.lower()

    for key in THREAT_LABELS:
        if key in label:
            return THREAT_LABELS[key]

    return 0.1  # default safe

def normalize_focus(focus_score: float) -> float:
    """
    Ensure focus is between 0 and 1
    """
    return max(0.0, min(1.0, focus_score))

def compute_threat_score(confidence: float, focus_score: float, label: str) -> float:
    """
    Combines:
    - Model confidence
    - Attention focus
    - Semantic threat of object
    """

    label_weight = get_label_threat_weight(label)
    focus_score = normalize_focus(focus_score)

    # Weighted fusion (IMPORTANT)
    threat_score = (
        0.4 * confidence +
        0.3 * focus_score +
        0.8 * label_weight
    ) / (0.4 + 0.3 + 0.8)

    return threat_score

def get_threat_level(threat_score: float) -> str:
    if threat_score > 0.7:
        return "HIGH"
    elif threat_score > 0.4:
        return "MEDIUM"
    else:
        return "SAFE"
