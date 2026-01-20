def compute_confidence(features: dict, reasoning: dict) -> float:
    score = 0.5

    if features["edge_density"] > 30:
        score += 0.2

    if features["brightness"] > 100:
        score += 0.1

    if "uncertain" in reasoning.get("attributes", []):
        score -= 0.2

    return round(min(max(score, 0.0), 1.0), 2)
