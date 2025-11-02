"""
Tokenizer following IIR best practices
Reference: Manning et al., Introduction to Information Retrieval
- Chapter 2: §2.2 (Tokenization)
- Chapter 2: §2.3 (Normalization, Stemming, Lemmatization)
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def tokenize(text, remove_stopwords=True, apply_stemming=True):
    """
    Tokenize text following IIR recommendations.
    
    Process:
    1. Tokenization (IIR §2.2.1): Split text into tokens using NLTK
    2. Normalization (IIR §2.2.3): Convert to lowercase
    3. Stopword removal (IIR §2.2.2): Remove common English stopwords by default
    4. Stemming (IIR §2.4): Apply Porter stemming algorithm by default
    
    Args:
        text: Input text to tokenize
        remove_stopwords: If True, remove common English stopwords (default: True)
        apply_stemming: If True, apply Porter stemming algorithm (default: True)
        
    Returns:
        List of processed tokens
    """
    if not text:
        return []
    
    # Step 1: Tokenization using NLTK (IIR §2.2.1)
    tokens = word_tokenize(text.lower())
    
    # Step 2: Keep only alphanumeric tokens (filter punctuation)
    tokens = [token for token in tokens if re.match(r'^[a-z0-9]+$', token)]
    
    # Step 3: Optional stopword removal (IIR §2.2.2)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]
    
    # Step 4: Optional stemming (IIR §2.4)
    if apply_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens