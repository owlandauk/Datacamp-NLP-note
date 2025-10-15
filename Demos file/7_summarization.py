
# ============================================
# 7_summarization.py
# --------------------------------------------
# Requirements: transformers
# Run: python 7_summarization.py
# ============================================
from transformers import pipeline
import os

os.makedirs("results", exist_ok=True)

text = (
    "Natural language processing (NLP) is a field of artificial intelligence that "
    "enables computers to understand, interpret, and generate human language."
)
summarizer = pipeline("summarization")
out_text = summarizer(text, max_length=30, min_length=10, do_sample=False)[0]["summary_text"]

with open("results/summarization.txt", "w", encoding="utf-8") as f:
    f.write(out_text)
print("Saved -> results/summarization.txt")
