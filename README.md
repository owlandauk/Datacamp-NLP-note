# 🧠 NLP Notes and Practical Demos

Natural Language Processing (NLP) helps computers work with human language.  
It covers a wide range of tasks — from text classification and translation to question answering and summarization.

This note combines **theory** and **practice**, showing how concepts evolve from Bag-of-Words to advanced Transformer models.  
Each demo can be run independently and saves its results in a `results/` folder.

---

## ⚙️ NLP Pipeline

A typical NLP pipeline consists of three stages:

1. **Preprocessing** — cleaning, tokenization, and normalization  
2. **Feature Extraction** — transforming text into numerical form (BoW, TF-IDF, embeddings)  
3. **Modeling** — applying algorithms (classical ML or Transformer-based deep learning)

---

## 🧱 Bag-of-Words (BoW)

- Represents each document as a vector of word counts  
- Ignores grammar and word order  
- Creates a simple numerical representation for textual data  

📘 **Demo:** [`1_bow_tfidf_heatmap.py`](nlp_demos/1_bow_tfidf_heatmap.py)  
- Saves: `results/bow_output.csv`, `results/tfidf_output.csv`, `results/tfidf_heatmap.png`  
- Libraries: `scikit-learn`, `matplotlib`

---

## ⚖️ From BoW to TF-IDF

While BoW counts how many times a word appears, **TF-IDF (Term Frequency–Inverse Document Frequency)** weighs how important that word is across documents.

- **TF (Term Frequency):** How often a word appears in a document  
- **IDF (Inverse Document Frequency):** How rare the word is in the entire corpus  
- **TF × IDF:** Highlights words that are frequent in one document but rare globally

📘 **Demo:** [`1_bow_tfidf_heatmap.py`](nlp_demos/1_bow_tfidf_heatmap.py)  
- Visualizes the TF-IDF weight matrix as a heatmap  
- Output example:  

  ![TF-IDF Heatmap](results/tfidf_heatmap.png)

---

## 🔡 Word Embeddings

Bag-of-Words treats all words as independent.  
**Word embeddings** map words to dense vectors that capture semantic meaning — similar words appear close in vector space.

- Example: `king - man + woman ≈ queen`
- Learned using models like Word2Vec, GloVe, or FastText

📘 **Demo:** [`2_word2vec_pca.py`](nlp_demos/2_word2vec_pca.py)  
- Trains a small Word2Vec model and visualizes embeddings using PCA  
- Output: `results/word2vec_pca.png`

---

## 🌐 GloVe Embeddings

**GloVe (Global Vectors for Word Representation)** is a pre-trained model from Wikipedia and Gigaword corpora.  
It captures both global co-occurrence statistics and local context information.

📘 **Demo 1:** [`3_glove_similarity.py`](nlp_demos/3_glove_similarity.py)  
- Finds top similar words to *“king”*  
- Output: `results/glove_similarity.txt`

📘 **Demo 2:** [`4_glove_pca.py`](nlp_demos/4_glove_pca.py)  
- Visualizes semantic clusters among words using PCA  
- Example word set: `["lion", "tiger", "leopard", "banana", "strawberry", "truck", "car", "bus"]`  
- Output: `results/glove_pca.png`

---

## 🧠 Transformers — Modern NLP

Transformers revolutionized NLP by introducing **self-attention** and contextual embeddings.  
Unlike Word2Vec or TF-IDF, Transformers understand *context* and *relationships* between words dynamically.

Common Transformer models:
- **BERT** — understanding language  
- **GPT** — generating text  
- **T5 / BART** — summarization, translation, and sequence-to-sequence learning

📦 Library: [Hugging Face Transformers](https://huggingface.co/transformers/)

---

## 🤖 Transformer Pipeline Demos

All scripts below use `transformers.pipeline()` for simplicity.  
Each saves its output into `results/`.

| # | Task | Script | Description | Example Output |
|:-:|------|---------|--------------|----------------|
| 5 | **Text Generation** | `5_text_generation.py` | Autocompletes text using GPT-2 | `text_generation.txt` |
| 6 | **Translation** | `6_translation_en_to_fr.py` | Translates English → French | `translation_en_fr.txt` |
| 7 | **Summarization** | `7_summarization.py` | Summarizes a paragraph | `summarization.txt` |
| 8 | **Question Answering** | `8_question_answering.py` | Extracts answer spans from text | `qa_result.json` |
| 9 | **Named Entity Recognition (NER)** | `9_named_entity_recognition.py` | Identifies names, organizations, and locations | `ner_result.json` |
| 10 | **Sentiment Analysis** | `10_sentiment_analysis.py` | Detects emotion polarity | `sentiment_result.json` |
| 11 | **Zero-shot Classification** | `11_zero_shot_classification.py` | Classifies unseen labels using natural language | `zero_shot_result.json` |

---

## 🧩 Specialized Transformer Tasks

| # | Task | Script | Model | Description |
|:-:|------|---------|--------|-------------|
| 12 | **Grammar Acceptability (CoLA)** | `12_cola_acceptability.py` | `textattack/bert-base-uncased-CoLA` | Judges if a sentence is grammatically correct |
| 13 | **Paraphrase Detection (QQP)** | `13_qqp_paraphrase.py` | `textattack/bert-base-uncased-QQP` | Checks if two questions mean the same thing |
| 14 | **Question Natural Language Inference (QNLI)** | `14_qnli_inference.py` | `textattack/bert-base-uncased-QNLI` | Determines if a passage answers a question |

---

## 🗣️ Linguistic Analysis — POS Tagging

Part-of-Speech (POS) tagging assigns grammatical roles (noun, verb, adjective, etc.) to each word.

📘 **Demo:** [`15_pos_tagging_spacy.py`](nlp_demos/15_pos_tagging_spacy.py)  
- Library: `spaCy`  
- Model: `en_core_web_sm`  
- Output: `results/pos_tags.txt`

---

## 📊 Visualization Examples

- **TF-IDF Heatmap:** Word importance across documents  
- **Word2Vec / GloVe PCA:** Semantic word clusters in 2D  
- **Transformer Outputs:** Saved as text/JSON for analysis

---

## ⚙️ Setup

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib gensim transformers spacy
python -m spacy download en_core_web_sm
