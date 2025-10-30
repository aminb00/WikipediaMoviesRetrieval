"""
Indexer - SPIMI In-Memory Inverted Index
Based on Manning et al. IIR Ch.4 Section 4.3
"""

from collections import defaultdict, Counter

# Index structures (term -> {doc_id: tf})
index = defaultdict(dict)

# Document terms (doc_id -> {term: tf}) - for efficient doc norm computation
doc_terms = defaultdict(dict)

# Metadata
doc_metadata = {}
doc_len = {}
term_cf = defaultdict(int)
collection_tokens = 0
num_documents = 0


def add_document(doc_id, tokens, metadata=None):
    """
    Add document to index using SPIMI algorithm.
    
    Args:
        doc_id: Document identifier
        tokens: List of tokens
        metadata: Dict with title, decade, plot
    """
    global num_documents, collection_tokens
    
    # Count term frequencies
    term_counts = Counter(tokens)
    
    # Update index: term -> {doc_id: tf}
    for term, tf in term_counts.items():
        index[term][doc_id] = tf
        doc_terms[doc_id][term] = tf
        term_cf[term] += tf
    
    # Store metadata and stats
    if metadata:
        doc_metadata[doc_id] = metadata
    
    doc_len[doc_id] = len(tokens)
    collection_tokens += len(tokens)
    num_documents += 1


def get_postings(term):
    """Get postings for term: {doc_id: tf}"""
    return index.get(term, {})


def get_document_frequency(term):
    """Get document frequency (number of docs containing term)"""
    return len(index.get(term, {}))


def get_collection_frequency(term):
    """Get collection frequency (total occurrences across all docs)"""
    return term_cf.get(term, 0)


def get_stats():
    """Get index statistics"""
    return {
        'num_documents': num_documents,
        'vocabulary_size': len(index),
        'collection_tokens': collection_tokens,
        'total_postings': sum(len(p) for p in index.values()),
        'avg_postings_per_term': sum(len(p) for p in index.values()) / len(index) if index else 0
    }


def build_index(documents_df):
    """
    Build inverted index from dataframe.
    
    Args:
        documents_df: DataFrame with columns: tokens, title, decade, plot
    """
    print("Building inverted index...")
    
    for idx, row in documents_df.iterrows():
        metadata = {
            'title': row['title'],
            'decade': row['decade'],
            'plot': row['plot']
        }
        
        add_document(idx, row['tokens'], metadata)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Indexed {idx + 1} documents...")
    
    print(f"  ✓ Indexed {num_documents} documents")
