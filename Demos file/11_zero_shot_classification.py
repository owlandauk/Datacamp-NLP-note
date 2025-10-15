
# ============================================
# 11_zero_shot_classification.py
# --------------------------------------------
# Requirements: transformers
# Run: python 11_zero_shot_classification.py
# ============================================
from transformers import pipeline
import os, json

os.makedirs("results", exist_ok=True)

classifier = pipeline("zero-shot-classification")
res = classifier(
    "This new phone has a great camera",
    candidate_labels=["technology", "sports", "politics"]
)

with open("results/zero_shot_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved -> results/zero_shot_result.json")
