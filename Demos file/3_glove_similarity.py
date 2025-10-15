
# ============================================
# 3_glove_similarity.py
# GloVe similarity lookup for a query word
# --------------------------------------------
# Requirements: gensim
# Run: python 3_glove_similarity.py
# ============================================
import gensim.downloader as api
import os

os.makedirs("results", exist_ok=True)

print("Loading GloVe (glove-wiki-gigaword-50)...")
model = api.load("glove-wiki-gigaword-50")
print("Loaded.")

query = "king"
topn = 10
sims = model.most_similar(query, topn=topn)

out = "results/glove_similarity.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(f"Top {topn} words similar to '{query}':\n")
    for w, s in sims:
        f.write(f"{w}\t{s:.4f}\n")
print(f"Saved similarity list -> {out}")
