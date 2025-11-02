import math
from collections import defaultdict
from Tokenizer import tokenize

# Logarithm base consistency: use natural log throughout
LOG = math.log

class QueryProcessor:
    def __init__(self, index_state, k1=1.5, b=0.75):
        """
        Initialize QueryProcessor with index state.
        
        IMPORTANT: Query tokenization must match document tokenization.
        Uses same settings: stemming enabled, stopwords kept.
        
        Args:
            index_state: Dictionary from Indexer.init_memory()
            k1: BM25 parameter (default 1.5)
            b: BM25 length normalization (default 0.75)
        """
        self.index = index_state["index"]
        self.doc_lengths = index_state["doc_len"]
        self.titles = index_state["titles"]
        self.N = len(self.doc_lengths)
        self.avgdl = sum(self.doc_lengths.values()) / self.N if self.N > 0 else 1
        self.k1 = k1
        self.b = b
        
        # Precompute per-doc max tf over ALL terms (not just query terms)
        # This is needed for augmented tf weighting
        self.doc_max_tf_all = defaultdict(int)
        for _, postings in self.index.items():
            for doc_id, tf in postings.items():
                if tf > self.doc_max_tf_all[doc_id]:
                    self.doc_max_tf_all[doc_id] = tf
        
        # Precompute collection term frequencies for language models
        # collection_tf[term] = total occurrences of term across all documents
        # collection_tokens = total tokens in collection
        self.collection_tf = defaultdict(int)
        for term, postings in self.index.items():
            self.collection_tf[term] = sum(postings.values())
        self.collection_tokens = sum(self.doc_lengths.values())
        
        self.doc_norms = {
            "lnc": self._compute_doc_norms("lnc"),
            "ltc": self._compute_doc_norms("ltc"),  
            "nnc": self._compute_doc_norms("nnc"), 
            "ntc": self._compute_doc_norms("ntc"),
            "anc": self._compute_doc_norms("anc"),
            "atc": self._compute_doc_norms("atc"),
        }

    def _compute_doc_norms(self, scheme):
        """
        Pre-compute document norms for a specific weighting scheme.
        SMART notation: tfn.idfn.normalization
        - tf: l=log, n=natural, a=augmented
        - idf: t=idf, n=none
        - norm: c=cosine, n=none
        """
        if len(scheme) != 3:
            raise ValueError(f"Invalid SMART scheme: {scheme}")
        
        tf_scheme = scheme[0]
        idf_scheme = scheme[1]
        norm_scheme = scheme[2]
        
        # If no cosine normalization, return empty dict
        if norm_scheme != 'c':
            return {}
        
        doc_norms = defaultdict(float)
        
        # Use precomputed doc_max_tf_all for augmented tf
        
        for term, postings in self.index.items():
            df = len(postings)
            if df == 0:
                continue
            
            idf = LOG(self.N / df) if idf_scheme == 't' else 1.0
            
            for doc_id, tf in postings.items():
                if tf <= 0:
                    continue
                
                if tf_scheme == 'l': 
                    tf_weight = 1 + LOG(tf)
                elif tf_scheme == 'n':
                    tf_weight = tf
                elif tf_scheme == 'a':
                    max_tf = self.doc_max_tf_all.get(doc_id, 0)
                    tf_weight = 0.5 + 0.5 * (tf / max_tf) if max_tf > 0 else 0
                else:
                    tf_weight = tf
                
                w_d = tf_weight * idf
                doc_norms[doc_id] += w_d ** 2
        
        # Cosine normalization
        for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])
        
        return dict(doc_norms)

    # -----------------------------
    # BM25 Retrieval
    # -----------------------------
    def compute_bm25_score(self, query):
        # Tokenize query with same settings as documents: stemming enabled, stopwords removed
        query_terms = tokenize(query, remove_stopwords=True, apply_stemming=True)
        scores = defaultdict(float)

        for term in query_terms:
            postings = self.index.get(term, {})
            df = len(postings)
            if df == 0:
                continue

            idf = LOG((self.N) / df)
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
    # SMART VSM Retrieval (Any variation)
    # -----------------------------
    def rank_smart(self, query, weighting="ltc.ltc", top_k=10):
        """
        Rank documents using SMART notation.
        
        Args:
            query: search query string
            weighting: SMART notation (e.g., "ltc.ltc", "lnc.ltc", "ntc.ntc")
                      Format: query_scheme.document_scheme
                      Each scheme: [tf][idf][norm]
                      - tf: l=log, n=natural, a=augmented
                      - idf: t=idf, n=none
                      - norm: c=cosine, n=none
            top_k: number of results to return
        """
        # Tokenize query with same settings as documents: stemming enabled, stopwords removed
        tokens = tokenize(query, remove_stopwords=True, apply_stemming=True)
        if not tokens:
            return []

        # Parse weighting scheme
        parts = weighting.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid SMART notation: {weighting}. Expected format: 'xxx.xxx'")
        
        query_scheme = parts[0]
        doc_scheme = parts[1]
        
        if len(query_scheme) != 3 or len(doc_scheme) != 3:
            raise ValueError(f"Invalid SMART notation: {weighting}. Each part must be 3 characters.")

        # Count query term frequencies
        query_tf = defaultdict(int)
        for t in tokens:
            query_tf[t] += 1

        # Find max tf in query for augmented normalization
        max_query_tf = max(query_tf.values()) if query_tf else 1

        # Compute query weights
        query_weights = {}
        for term, tf in query_tf.items():
            df = len(self.index.get(term, {}))
            if df == 0:
                continue
            
            if query_scheme[0] == 'l':
                tf_weight = 1 + LOG(tf)
            elif query_scheme[0] == 'n':
                tf_weight = tf
            elif query_scheme[0] == 'a': 
                tf_weight = 0.5 + 0.5 * (tf / max_query_tf)
            else:
                tf_weight = tf
            
            if query_scheme[1] == 't':
                idf = LOG(self.N / df)
                tf_weight *= idf
            
            query_weights[term] = tf_weight

        if query_scheme[2] == 'c':
            norm_q = math.sqrt(sum(w ** 2 for w in query_weights.values()))
            if norm_q > 0:
                for term in query_weights:
                    query_weights[term] /= norm_q

        # Use precomputed doc_max_tf_all for augmented tf (not query-only max tf)

        # Score documents
        scores = defaultdict(float)

        for term, w_q in query_weights.items():
            postings = self.index.get(term, {})
            df = len(postings)
            if df == 0:
                continue
            
            idf = LOG(self.N / df) if doc_scheme[1] == 't' else 1.0
            
            for doc_id, tf in postings.items():
                if tf <= 0:
                    continue
                
                if doc_scheme[0] == 'l':
                    tf_weight = 1 + LOG(tf)
                elif doc_scheme[0] == 'n': 
                    tf_weight = tf
                elif doc_scheme[0] == 'a':
                    max_tf = self.doc_max_tf_all.get(doc_id, 0)
                    tf_weight = 0.5 + 0.5 * (tf / max_tf) if max_tf > 0 else 0
                else:
                    tf_weight = tf
                
                w_d = tf_weight * idf
                
                scores[doc_id] += w_q * w_d

        # Apply cosine normalization using precomputed norms
        if doc_scheme[2] == 'c':
            if doc_scheme not in self.doc_norms:
                raise ValueError(f"Document scheme '{doc_scheme}' not precomputed. Supported schemes: {list(self.doc_norms.keys())}")
            for doc_id in scores:
                norm_d = self.doc_norms[doc_scheme].get(doc_id, 1.0)
                if norm_d > 0:
                    scores[doc_id] /= norm_d

        # Rank and return results
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.titles[doc_id], score) for doc_id, score in ranked[:top_k]]

    # -----------------------------
    # Language Model Retrieval (Dirichlet Prior Smoothing)
    # -----------------------------
    def compute_lm_score(self, query, mu=2000, top_k=10):
        """
        Rank documents using Language Model with Dirichlet prior smoothing.
        
        Formula: P(q|d) = sum_{t in q} log((tf_{t,d} + μ * P(t|C)) / (|d| + μ))
        where:
        - tf_{t,d} = term frequency of term t in document d
        - P(t|C) = collection probability = collection_tf[t] / collection_tokens
        - |d| = document length
        - μ = Dirichlet smoothing parameter (default: 2000)
        
        Args:
            query: search query string
            mu: Dirichlet smoothing parameter (default: 2000)
            top_k: number of results to return
            
        Returns:
            List of (title, score) tuples, sorted by score descending
        """
        # Tokenize query with same settings as documents: stemming enabled, stopwords removed
        query_terms = tokenize(query, remove_stopwords=True, apply_stemming=True)
        if not query_terms:
            return []
        
        # Compute collection probabilities P(t|C) for query terms
        # P(t|C) = collection_tf[t] / collection_tokens
        collection_probs = {}
        for term in query_terms:
            if term in self.collection_tf:
                collection_probs[term] = self.collection_tf[term] / self.collection_tokens
            else:
                # Term not in collection: use small epsilon to avoid log(0)
                collection_probs[term] = 1.0 / self.collection_tokens if self.collection_tokens > 0 else 1e-10
        
        # Collect candidate documents (documents containing at least one query term)
        candidate_docs = set()
        term_postings = {}
        for term in query_terms:
            postings = self.index.get(term, {})
            if postings:
                term_postings[term] = postings
                candidate_docs.update(postings.keys())
        
        # Score documents
        # For each candidate document, sum log probabilities over ALL query terms
        scores = defaultdict(float)
        
        for doc_id in candidate_docs:
            doc_length = self.doc_lengths.get(doc_id, 0)
            if doc_length == 0:
                continue
            
            # Sum log probabilities for all query terms
            for term in query_terms:
                # Get term frequency in document (0 if term doesn't appear)
                tf = term_postings.get(term, {}).get(doc_id, 0)
                P_t_C = collection_probs[term]
                
                # Dirichlet smoothing formula
                numerator = tf + mu * P_t_C
                denominator = doc_length + mu
                
                # Add log probability for this term-document pair
                if numerator > 0 and denominator > 0:
                    scores[doc_id] += LOG(numerator / denominator)
        
        # Sort results
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Return doc titles + scores
        return [(self.titles[doc_id], score) for doc_id, score in ranked[:top_k]]