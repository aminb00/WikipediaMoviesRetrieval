"""
Indexer Component - Builds inverted index from documents

An inverted index maps each term to the list of documents containing it.
Structure: {term: [(doc_id, frequency), (doc_id, frequency), ...]}

This is the most efficient data structure for document retrieval.
"""

from collections import defaultdict, Counter

class InvertedIndex:
    """
    Inverted index data structure for efficient document retrieval.
    Maps terms to posting lists (document IDs and term frequencies).
    """
    
    def __init__(self):
        # Main index: term -> list of (doc_id, term_frequency)
        self.index = defaultdict(list)
        
        # Document metadata
        self.doc_metadata = {}  # doc_id -> {title, decade, etc}
        
        # Statistics
        self.num_documents = 0
        self.vocabulary = set()
        
    def add_document(self, doc_id, tokens, metadata=None):
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            tokens: List of tokens from the document
            metadata: Optional dict with document metadata (title, decade, etc)
        """
        # Count term frequencies in document
        term_freq = Counter(tokens)
        
        # Add to inverted index
        for term, freq in term_freq.items():
            self.index[term].append((doc_id, freq))
            self.vocabulary.add(term)
        
        # Store metadata
        if metadata:
            self.doc_metadata[doc_id] = metadata
        
        self.num_documents += 1
    
    def get_postings(self, term):
        """
        Get posting list for a term.
        
        Args:
            term: The term to look up
            
        Returns:
            List of (doc_id, frequency) tuples, or empty list if term not found
        """
        return self.index.get(term, [])
    
    def get_document_frequency(self, term):
        """
        Get document frequency (number of documents containing the term).
        
        Args:
            term: The term to look up
            
        Returns:
            Number of documents containing the term
        """
        return len(self.index.get(term, []))
    
    def get_stats(self):
        """
        Get index statistics.
        
        Returns:
            Dict with index statistics
        """
        total_postings = sum(len(postings) for postings in self.index.values())
        
        return {
            'num_documents': self.num_documents,
            'vocabulary_size': len(self.vocabulary),
            'total_postings': total_postings,
            'avg_postings_per_term': total_postings / len(self.vocabulary) if self.vocabulary else 0
        }
    
    def __len__(self):
        """Return number of unique terms in index."""
        return len(self.vocabulary)


def build_index(documents_df):
    """
    Build inverted index from a dataframe of documents.
    
    Args:
        documents_df: DataFrame with columns: tokens, title, decade, etc.
        
    Returns:
        InvertedIndex object
    """
    print("Building inverted index...")
    
    index = InvertedIndex()
    
    # Index each document
    for idx, row in documents_df.iterrows():
        doc_id = idx  # Use dataframe index as doc_id
        tokens = row['tokens']
        
        # Store metadata
        metadata = {
            'title': row['title'],
            'decade': row['decade'],
            'plot': row['plot']
        }
        
        index.add_document(doc_id, tokens, metadata)
        
        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"  Indexed {idx + 1} documents...")
    
    print(f"  ✓ Indexed {index.num_documents} documents")
    
    return index
