import math
from collections import defaultdict
from Tokenizer import tokenize

class QueryProcessor:
    def __init__(self, index_state, k1=1.5, b=0.75):
        """
        index_state: the dictionary returned by Indexer.init_memory() and built by Indexer.index_doc_mem()
        """
        self.index = index_state["index"]
        self.doc_lengths = index_state["doc_len"]
        self.titles = index_state["titles"]
        self.N = len(self.doc_lengths)
        self.avgdl = sum(self.doc_lengths.values()) / self.N
        self.k1 = k1
        self.b = b

    # -----------------------------
    # BM25 Retrieval
    # -----------------------------
    def compute_bm25_score(self, query):
        query_terms = tokenize(query)
        scores = defaultdict(float)

        for term in query_terms:
            postings = self.index.get(term, {})
            df = len(postings)
            if df == 0:
                continue

            idf = math.log((self.N) / df)
            for doc_id, tf in postings.items():
                dl = self.doc_lengths.get(doc_id, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] += score

        # Sort results
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Return doc titles + scores
        return [(self.titles[doc_id], score) for doc_id, score in ranked[:10]]

    # -----------------------------
    # SMART ltc.ltc Retrieval
    # -----------------------------
    def rank_smart(self, query, weighting="ltc.ltc", top_k=10):
        tokens = tokenize(query)
        if not tokens:
            return []

        query_tf = defaultdict(int)
        for t in tokens:
            query_tf[t] += 1

        # Query weighting
        query_weights = {}
        for term, tf in query_tf.items():
            df = len(self.index.get(term, {}))
            if df == 0:
                continue
            idf = math.log(self.N / df)
            if weighting.startswith("ltc"):
                w_q = (1 + math.log(tf)) * idf
            elif weighting.startswith("ntc"):
                w_q = tf * idf
            query_weights[term] = w_q

        # Normalize query vector
        norm_q = math.sqrt(sum(w ** 2 for w in query_weights.values()))
        for term in query_weights:
            query_weights[term] /= norm_q

        # Score each document
        scores = defaultdict(float)
        for term, w_q in query_weights.items():
            postings = self.index.get(term, {})
            for doc_id, tf in postings.items():
                w_d = 1 + math.log(tf)
                scores[doc_id] += w_q * w_d

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.titles[doc_id], score) for doc_id, score in ranked[:top_k]]
