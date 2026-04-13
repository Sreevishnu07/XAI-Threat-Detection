from app.model.threat import get_label_threat_weight


def generate_explanation(label, confidence, focus_scores, consistency, threat_level, uncertainty):
    
    avg_focus = sum(focus_scores.values()) / len(focus_scores)

    label_weight = get_label_threat_weight(label)

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

    if label_weight < 0.2:
        threat_text = "indicating that the object is safe and not a threat"
    elif threat_level == "HIGH":
        threat_text = "indicating a high-risk object"
    elif threat_level == "MEDIUM":
        threat_text = "suggesting moderate risk"
    else:
        threat_text = "indicating a safe object"

    explanation = (
        f"The model predicts '{label}' with {conf_text}. "
        f"It shows {focus_text}, with {consistency_text}, "
        f"{threat_text}."
    )

    if uncertainty == "WARNING":
        explanation += " However, the explanation methods show low agreement, so this prediction should be interpreted with caution."
    elif uncertainty == "LOW_RISK_OBJECT":
        explanation += " Although the model is highly confident, the object belongs to a low-risk category."
    elif uncertainty == "LOW_CONFIDENCE":
        explanation += " The model confidence is low, so this prediction may not be reliable."

    return explanation
