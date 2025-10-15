
# ============================================
# 10_sentiment_analysis.py
# --------------------------------------------
# Requirements: transformers
# Run: python 10_sentiment_analysis.py
# ============================================
from transformers import pipeline
import os, json

os.makedirs("results", exist_ok=True)

sentiment = pipeline("sentiment-analysis")
res = sentiment("I love working with NLP!")

with open("results/sentiment_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved -> results/sentiment_result.json")
