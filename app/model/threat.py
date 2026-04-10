import numpy as np

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

    weights = []
    for key, val in THREAT_LABELS.items():
        if key in label:
            weights.append(val)

    return max(weights) if weights else 0.1


def normalize_focus(focus_score: float) -> float:
    return float(np.clip(focus_score, 0.0, 1.0))


def compute_consistency(focus_scores: dict) -> float:
    values = list(focus_scores.values())

    if len(values) < 2:
        return 0.5

    std_dev = np.std(values)

    consistency = 1.0 - std_dev

    return float(np.clip(consistency, 0.0, 1.0))


def get_trust_level(consistency: float) -> str:
    if consistency > 0.75:
        return "HIGH"
    elif consistency > 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def compute_threat_score(confidence: float, focus_scores: dict, label: str):
    """
    Returns:
    threat_score, consistency
    """

    label_weight = get_label_threat_weight(label)

    focus_values = [normalize_focus(f) for f in focus_scores.values()]
    avg_focus = np.mean(focus_values)

    consistency = compute_consistency(focus_scores)

    threat_score = (
        0.4 * confidence +
        0.2 * avg_focus +
        0.25 * label_weight +
        0.15 * consistency
    )

    threat_score = float(np.clip(threat_score, 0.0, 1.0))

    return threat_score, consistency

def get_threat_level(threat_score: float) -> str:
    if threat_score > 0.75:
        return "HIGH"
    elif threat_score > 0.45:
        return "MEDIUM"
    else:
        return "SAFE"
