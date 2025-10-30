# 🎬 WikipediaMoviesRetrieval  
**Information Retrieval – Assignment 1 (2025/2026)**  
**Prof. Toon Calders – University of Antwerp**

---

## 🧠 Overview
This project implements from scratch a **document search and ranked retrieval system**, following the concepts taught in the *Information Retrieval* course.

The system builds an **inverted index** for a collection of documents and ranks them according to their **relevance** to a given query, using the **Vector Space Model (VSM)** with the **SMART ltc.ltc** weighting scheme.  

The dataset used for this project is the [Wikipedia Movies Dataset](https://www.kaggle.com/datasets/exactful/wikipedia-movies), containing metadata and textual descriptions of thousands of films.

No external search libraries (Lucene, Solr, ElasticSearch, etc.) are used — all core components (tokenizer, indexer, query processor) are **self-implemented**.

---

## 🧩 Components Implemented
| Component | Description | Points |
|------------|--------------|--------|
| **Tokenizer** | Translates documents into normalized tokens (lowercasing, punctuation removal, stopword filtering). | 3 |
| **Indexer (in-memory)** | Builds a full in-memory inverted index from all documents. | 5 |
| **Query Processor** | Processes queries and ranks documents using cosine similarity with SMART `ltc.ltc`. | 4 |
| **Total (core)** | **12 / 20** (full core implementation) |   |
| **Extensions (optional)** | *To be added if implemented (e.g., BM25, phrase queries, spell-check, etc.)* |   |

---

## 📂 Repository Structure
