
# ============================================
# 2_word2vec_pca.py
# Train tiny Word2Vec and visualize via PCA
# --------------------------------------------
# Requirements: gensim, scikit-learn, matplotlib
# Run: python 2_word2vec_pca.py
# ============================================
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

sentences = [
    ["NLP", "is", "fun"],
    ["I", "love", "machine", "learning"],
    ["deep", "learning", "for", "NLP"]
]

model = Word2Vec(sentences, vector_size=20, min_count=1, epochs=100)

words = list(model.wv.index_to_key)
X = model.wv[words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (result[i, 0] + 0.03, result[i, 1] + 0.03))
plt.title("Word2Vec Embeddings (2D PCA)")
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
out = "results/word2vec_pca.png"
plt.savefig(out); plt.close()
print(f"Saved PCA plot -> {out}")
