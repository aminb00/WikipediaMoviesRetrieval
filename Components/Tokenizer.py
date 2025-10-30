"""
Tokenizer for document retrieval system using NLTK.
Converts text into tokens (normalized words).

Tokenization steps:
1. Tokenize using NLTK word_tokenize
2. Lowercase conversion
3. Remove punctuation and non-alphabetic tokens
4. Remove stopwords (using NLTK's stopwords)
5. Optional: Stemming with Porter Stemmer (LEMMATIZATION CAN BE ADDED LATER)
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (only runs once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load English stopwords
STOPWORDS = set(stopwords.words('english'))

# Initialize stemmer
stemmer = PorterStemmer()

def tokenize(text, use_stemming=True):
    """
    Convert text into list of tokens using NLTK.
    
    Args:
        text (str): Input text to tokenize
        use_stemming (bool): Whether to apply stemming (default: True)
        
    Returns:
        list: List of normalized tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # Step 1: Tokenize with NLTK
    tokens = word_tokenize(text.lower())
    
    # Step 2: Keep only alphabetic tokens (removes punctuation, numbers)
    tokens = [t for t in tokens if t.isalpha()]
    
    # Step 3: Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    
    # Step 4: Optional stemming (reduces words to root form)
    if use_stemming:
        tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens


