
# ============================================
# 13_qqp_paraphrase.py
# --------------------------------------------
# Requirements: transformers
# Run: python 13_qqp_paraphrase.py
# ============================================
from transformers import pipeline
import os, json

os.makedirs("results", exist_ok=True)

model = "textattack/bert-base-uncased-QQP"
clf = pipeline("text-classification", model=model)
res = clf([
    "How can I learn NLP?",
    "What is the best way to study natural language processing?"
])

with open("results/qqp_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved -> results/qqp_result.json")
