"""
Tokenizer (minimal, clean)
Reference: IIR Ch.2 §2.2-2.3
"""

import re

def tokenize(text):
    """Lowercase + alphanumeric tokens only."""
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())