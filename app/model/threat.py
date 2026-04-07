def normalize_focus(focus_score: float) -> float:
    return focus_score / (focus_score + 1)


def compute_threat_score(confidence: float, focus_score: float) -> float:
    normalized_focus = normalize_focus(focus_score)
    return confidence * normalized_focus


def get_threat_level(threat_score: float) -> str:
    if threat_score > 0.6:
        return "HIGH"
    elif threat_score > 0.3:
        return "MEDIUM"
    else:
        return "SAFE"
