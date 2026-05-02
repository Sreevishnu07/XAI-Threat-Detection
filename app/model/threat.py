import numpy as np
import torch

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()

    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    )


def compute_consistency_maps(cam_pp_map: np.ndarray,
                             score_map: np.ndarray,
                             ig_map: np.ndarray) -> float:
    def normalize(x):
        x = x - x.min()
        return x / (x.max() + 1e-8)

    cam_pp_map = normalize(cam_pp_map)
    score_map = normalize(score_map)
    ig_map = normalize(ig_map)

    sim1 = cosine_similarity(cam_pp_map, score_map)
    sim2 = cosine_similarity(cam_pp_map, ig_map)
    sim3 = cosine_similarity(score_map, ig_map)

    consistency = (sim1 + sim2 + sim3) / 3.0

    return float(np.clip(consistency, 0.0, 1.0))


def get_trust_level(consistency: float) -> str:
    if consistency > 0.75:
        return "HIGH"
    elif consistency > 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def compute_threat_score(confidence: float,
                         focus_scores: dict,
                         label: str,
                         cam_pp_map: np.ndarray,
                         score_map: np.ndarray,
                         ig_map: np.ndarray):

    label_weight = get_label_threat_weight(label)

    focus_values = [normalize_focus(f) for f in focus_scores.values()]
    avg_focus = np.mean(focus_values)

    consistency = compute_consistency_maps(cam_pp_map, score_map, ig_map)

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


def compute_uncertainty(confidence: float, consistency: float, label: str) -> str:
    label_weight = get_label_threat_weight(label)

    if confidence > 0.85 and consistency < 0.5:
        return "WARNING"

    if confidence > 0.85 and label_weight < 0.2:
        return "LOW_RISK_OBJECT"

    if confidence < 0.5:
        return "LOW_CONFIDENCE"

    return "NORMAL"



def smoothgrad_integrated_gradients(model,
                                     input_tensor,
                                     generate_ig_fn,
                                     n_samples=15,
                                     noise_sigma=0.05):

    model.eval()
    ig_maps = []

    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * noise_sigma
        noisy_input = input_tensor + noise

        # keep valid pixel range
        noisy_input = torch.clamp(noisy_input, 0, 1)

        ig = generate_ig_fn(model, noisy_input)  # your existing IG function

        if isinstance(ig, torch.Tensor):
            ig = ig.detach().cpu().numpy()

        ig_maps.append(ig)

    ig_maps = np.stack(ig_maps, axis=0)

    return np.mean(ig_maps, axis=0)


def normalize_ig_map(ig_map: np.ndarray) -> np.ndarray:
    low = np.percentile(ig_map, 3)
    high = np.percentile(ig_map, 97)

    ig_map = np.clip((ig_map - low) / (high - low + 1e-8), 0, 1)

    ig_map = np.power(ig_map, 1.5)

    return ig_map
