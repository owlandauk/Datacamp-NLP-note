
# ============================================
# 15_pos_tagging_spacy.py
# --------------------------------------------
# Requirements: spacy (and model: en_core_web_sm)
# Run: python -m spacy download en_core_web_sm
#      python 15_pos_tagging_spacy.py
# ============================================
import spacy, os

os.makedirs("results", exist_ok=True)

nlp = spacy.load("en_core_web_sm")
doc = nlp("NLP lets computers understand natural language.")

out = "\n".join([f"{t.text}\t{t.pos_}\t{t.tag_}" for t in doc])
with open("results/pos_tags.txt", "w", encoding="utf-8") as f:
    f.write(out)
print("Saved -> results/pos_tags.txt")
