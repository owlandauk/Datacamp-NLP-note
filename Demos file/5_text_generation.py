
# ============================================
# 5_text_generation.py
# --------------------------------------------
# Requirements: transformers
# Run: python 5_text_generation.py
# ============================================
from transformers import pipeline
import os

os.makedirs("results", exist_ok=True)

generator = pipeline("text-generation", model="gpt2")
out_text = generator("Natural language processing is", max_length=40)[0]["generated_text"]

with open("results/text_generation.txt", "w", encoding="utf-8") as f:
    f.write(out_text)
print("Saved -> results/text_generation.txt")
