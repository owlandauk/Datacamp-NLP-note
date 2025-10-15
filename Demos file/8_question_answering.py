
# ============================================
# 8_question_answering.py
# --------------------------------------------
# Requirements: transformers
# Run: python 8_question_answering.py
# ============================================
from transformers import pipeline
import os, json

os.makedirs("results", exist_ok=True)

qa = pipeline("question-answering")
context = "Transformers are powerful models introduced in 2017 by researchers at Google."
res = qa(question="Who introduced Transformers?", context=context)

with open("results/qa_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved -> results/qa_result.json")
