
# ============================================
# 6_translation_en_to_fr.py
# --------------------------------------------
# Requirements: transformers
# Run: python 6_translation_en_to_fr.py
# ============================================
from transformers import pipeline
import os

os.makedirs("results", exist_ok=True)

translator = pipeline("translation_en_to_fr")
out_text = translator("NLP is amazing!", max_length=40)[0]["translation_text"]

with open("results/translation_en_fr.txt", "w", encoding="utf-8") as f:
    f.write(out_text)
print("Saved -> results/translation_en_fr.txt")
