
# ============================================
# 4_glove_pca.py
# Visualize selected GloVe vectors via PCA
# --------------------------------------------
# Requirements: gensim, scikit-learn, matplotlib
# Run: python 4_glove_pca.py
# ============================================
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

print("Loading GloVe (glove-wiki-gigaword-50)...")
model = api.load("glove-wiki-gigaword-50")
print("Loaded.")

words = ["lion", "tiger", "leopard", "banana", "strawberry", "truck", "car", "bus"]

vectors = [model[w] for w in words]
pca = PCA(n_components=2)
vec2d = pca.fit_transform(vectors)

plt.figure(figsize=(7, 7))
plt.scatter(vec2d[:, 0], vec2d[:, 1])
for w, (x, y) in zip(words, vec2d):
    plt.annotate(w, (x + 0.05, y + 0.05))
plt.title("GloVe Wikipedia Word Embeddings (2D PCA)")
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
out = "results/glove_pca.png"
plt.savefig(out); plt.close()
print(f"Saved PCA plot -> {out}")
