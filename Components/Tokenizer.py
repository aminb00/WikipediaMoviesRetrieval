"""
Simple Tokenizer (Functional, Minimal)
Reference: Manning et al. IIR Ch.2 §2.2.1-2.2.3

Features:
- Case folding (lowercase)
- Diacritics removal
- Optional stopword removal
"""

import re
import unicodedata

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
    _spacy_available = True
except Exception:
    spacy = None
    SPACY_STOP_WORDS = set()
    _spacy_available = False

_nlp = None
if _spacy_available:
    try:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "senter", "textcat"])
    except Exception:
        _nlp = None


def _strip_diacritics(text: str):
    """Remove diacritics. Reference: IIR Ch.2 §2.2.1."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _normalize(text: str):
    """Normalize text: lowercase + diacritics removal."""
    return _strip_diacritics(text.lower())


class Tokenizer:
    """Simple tokenizer with normalization and optional stopword removal."""
    
    def __init__(self, remove_stopwords: bool = False):
        """Initialize tokenizer.
        
        Args:
            remove_stopwords: Remove stopwords if True (default: False)
        """
        self.remove_stopwords = remove_stopwords
    
    def tokenize(self, text: str):
        """Tokenize text. Returns list of tokens."""
        if not text or not isinstance(text, str):
            return []
        
        text_norm = _normalize(text)
        
        # Use spaCy if available
        if _nlp:
            doc = _nlp(text_norm)
            tokens = []
            for t in doc:
                if t.is_space or t.is_punct:
                    continue
                tok = t.lemma_.lower() if t.lemma_ != "-PRON-" else t.text.lower()
                if not self.remove_stopwords or tok not in SPACY_STOP_WORDS:
                    tokens.append(tok)
            return tokens
        
        # Fallback: simple regex tokenization
        tokens = re.findall(r"[a-z0-9]+", text_norm)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in SPACY_STOP_WORDS]
        return tokens
    
    def __call__(self, text: str):
        """Shortcut: tokenizer(text) -> tokenize(text)."""
        return self.tokenize(text)
    
    def __repr__(self):
        stopwords_status = "on" if self.remove_stopwords else "off"
        spacy_status = "on" if _nlp else "off"
        return f"Tokenizer(spacy={spacy_status}, stopwords={stopwords_status})"


if __name__ == "__main__":
    tk = Tokenizer(remove_stopwords=False)
    s = "Information Retrieval is fun. Tokenization processes text."
    print(tk)
    print(tk.tokenize(s))