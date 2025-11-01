import math
from collections import defaultdict
from Tokenizer import Tokenizer

class QueryProcessor:
    def __init__(self, index_state):
        """
        index_state: dictionary returned by Indexer.init_memory()
        """
        self.index = index_state["index"]
        self.doc_lengths = index_state["doc_len"]
        self.titles = index_state["titles"]
        self.N = len(self.doc_lengths)
        self.tk = Tokenizer()

    # -----------------------------
    # SMART ltc.ltc Retrieval
    # -----------------------------
    def rank_smart(self, query, weighting="ltc.ltc", top_k=10):
        tokens = self.tk.tokenize(query)
        if not tokens:
            return []

        query_tf = defaultdict(int)
        for t in tokens:
            query_tf[t] += 1

        # Compute query weights (ltc)
        query_weights = {}
        for term, tf in query_tf.items():
            df = len(self.index.get(term, {}))
            if df == 0:
                continue
            idf = math.log(self.N / df)
            w_q = (1 + math.log(tf)) * idf
            query_weights[term] = w_q

        # Normalize query vector
        norm_q = math.sqrt(sum(w ** 2 for w in query_weights.values()))
        for term in query_weights:
            query_weights[term] /= norm_q

        # Compute document scores
        scores = defaultdict(float)
        for term, w_q in query_weights.items():
            postings = self.index.get(term, {})
            for doc_id, tf in postings.items():
                w_d = 1 + math.log(tf)
                scores[doc_id] += w_q * w_d

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.titles[doc_id], score) for doc_id, score in ranked[:top_k]]
