
# ============================================
# 9_named_entity_recognition.py
# --------------------------------------------
# Requirements: transformers
# Run: python 9_named_entity_recognition.py
# ============================================
from transformers import pipeline
import os, json

os.makedirs("results", exist_ok=True)

ner = pipeline("ner", grouped_entities=True)
res = ner("Hugging Face Inc. is a company based in New York City.")

with open("results/ner_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved -> results/ner_result.json")
