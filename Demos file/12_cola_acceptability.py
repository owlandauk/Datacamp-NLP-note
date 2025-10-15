
# ============================================
# 12_cola_acceptability.py
# --------------------------------------------
# Requirements: transformers
# Run: python 12_cola_acceptability.py
# ============================================
from transformers import pipeline
import os, json

os.makedirs("results", exist_ok=True)

cola = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")
res = cola("This sentence are ungrammatical.")

with open("results/cola_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved -> results/cola_result.json")
