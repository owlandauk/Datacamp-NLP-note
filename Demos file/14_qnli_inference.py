
# ============================================
# 14_qnli_inference.py
# --------------------------------------------
# Requirements: transformers
# Run: python 14_qnli_inference.py
# ============================================
from transformers import pipeline
import os, json

os.makedirs("results", exist_ok=True)

model = "textattack/bert-base-uncased-QNLI"
qnli = pipeline("text-classification", model=model)
res = qnli({ "question": "What is NLP?", "text": "NLP stands for Natural Language Processing." })

with open("results/qnli_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved -> results/qnli_result.json")
