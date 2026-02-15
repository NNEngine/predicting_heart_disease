import json

THRESHOLD = 0.80  # required F1 score

with open("metrics.json") as f:
    metrics = json.load(f)

if metrics["f1_score"] < THRESHOLD:
    raise ValueError(
        f"Model rejected: f1_score={metrics['f1_score']} below threshold {THRESHOLD}"
    )

print("Model passed evaluation gate")
