def generate_explanation(label, confidence, focus_scores, consistency, threat_level):
    
    avg_focus = sum(focus_scores.values()) / len(focus_scores)

    if confidence > 0.85:
        conf_text = "high confidence"
    elif confidence > 0.6:
        conf_text = "moderate confidence"
    else:
        conf_text = "low confidence"

    if avg_focus > 0.6:
        focus_text = "strong focus on key object regions"
    elif avg_focus > 0.4:
        focus_text = "moderate attention on relevant areas"
    else:
        focus_text = "diffused attention across the image"

    if consistency > 0.75:
        consistency_text = "high agreement across explanation methods"
    elif consistency > 0.5:
        consistency_text = "moderate agreement across methods"
    else:
        consistency_text = "low agreement across methods"

    if threat_level == "HIGH":
    threat_text = "indicating a high-risk object"
    elif threat_level == "MEDIUM":
    threat_text = "suggesting moderate risk"
    else:
    threat_text = "indicating a safe and non-threatening object"

    explanation = (
        f"The model predicts '{label}' with {conf_text}. "
        f"It shows {focus_text}, with {consistency_text}, "
        f"{threat_text}."
    )

    return explanation
