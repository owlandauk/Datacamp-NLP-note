
# ============================================
# 1_bow_tfidf_heatmap.py
# Bag-of-Words & TF-IDF with a simple heatmap
# --------------------------------------------
# Requirements: scikit-learn, pandas, matplotlib
# Run: python 1_bow_tfidf_heatmap.py
# ============================================
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

corpus = [
    "I love NLP and machine learning",
    "NLP is fun and powerful",
    "I love learning new things"
]

# Bag-of-Words
bow = CountVectorizer()
X_bow = bow.fit_transform(corpus)
df_bow = pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names_out())
df_bow.to_csv("results/bow_output.csv", index=False)
print("Saved Bag-of-Words matrix -> results/bow_output.csv")

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
df_tfidf.to_csv("results/tfidf_output.csv", index=False)
print("Saved TF-IDF matrix       -> results/tfidf_output.csv")

# Heatmap (matplotlib only; no seaborn)
plt.figure(figsize=(8, 3))
plt.imshow(df_tfidf.values, aspect="auto")
plt.xticks(range(len(df_tfidf.columns)), df_tfidf.columns, rotation=90)
plt.yticks(range(len(corpus)), [f"doc{i+1}" for i in range(len(corpus))])
plt.title("TF-IDF Heatmap")
plt.colorbar()
plt.tight_layout()
plt.savefig("results/tfidf_heatmap.png")
plt.close()
print("Saved TF-IDF heatmap      -> results/tfidf_heatmap.png")
