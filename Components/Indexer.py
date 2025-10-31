"""
Tiny IR Indexer (Functional, Minimal)
(a) Memory SPIMI (5 pts)  (b) Disk term-per-file (+2)  (c) Updatable aux+merge (+2)

References: Manning et al. IIR
- SPIMI: IIR §4.3
- Disk-based: IIR §5.2
- Dynamic Indexing: IIR §4.5
"""

import os
import re
import pickle
import hashlib
from collections import defaultdict, Counter
from typing import Dict, Tuple, Callable

try:
    from tokenizer import Tokenizer
except ImportError:
    Tokenizer = None


# --- Common ---
def tokenize(text: str, tokenizer=None):
    """Simple tokenizer (lowercase, alphanumeric). Use Tokenizer if provided."""
    if tokenizer and hasattr(tokenizer, 'tokenize'):
        return tokenizer.tokenize(text)
    return re.findall(r"[a-z0-9]+", text.lower())


def read_txt(path: str) -> Tuple[str, str]:
    """Read text file: returns (title, content)."""
    with open(path, "r", encoding="utf-8") as f:
        return os.path.basename(path), f.read()


# =============================================================================
# (a) MEMORY (RAM SPIMI)
# Reference: IIR §4.3
# =============================================================================
def init_memory(tokenizer=None):
    """Initialize memory-resident indexer state."""
    return {
        "index": defaultdict(dict),  # term -> {doc_id: tf}
        "doc_len": {},
        "titles": {},
        "next_id": 0,
        "tokenizer": tokenizer
    }


def index_doc_mem(st, title: str, text: str):
    """Index a single document using SPIMI algorithm."""
    did = st["next_id"]
    st["next_id"] += 1
    toks = tokenize(text, st.get("tokenizer"))
    st["doc_len"][did] = len(toks)
    st["titles"][did] = title
    for t, tf in Counter(toks).items():
        st["index"][t][did] = tf
    return did


def build_memory(st, folder: str, read_fn: Callable[[str], Tuple[str, str]] = read_txt):
    """Build memory index from folder of documents."""
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            title, text = read_fn(p)
            index_doc_mem(st, title, text)


def postings_mem(st, term: str) -> Dict[int, int]:
    """Get postings list for a term: {doc_id: tf}."""
    return dict(st["index"].get(term, {}))


# =============================================================================
# (b) DISK (term-per-file, only relevant parts loaded)
# Reference: IIR §5.2
# =============================================================================
def init_disk(index_dir="index", tokenizer=None):
    """Initialize disk-based indexer state."""
    os.makedirs(os.path.join(index_dir, "terms"), exist_ok=True)
    return {
        "dir": index_dir,
        "lex": {},  # term -> {"path": path, "df": int, "cf": int}
        "doc_len": {},
        "titles": {},
        "next_id": 0,
        "tokenizer": tokenizer
    }


def _term_path(index_dir: str, term: str) -> str:
    """Generate file path for term's postings using MD5 hash."""
    h = hashlib.md5(term.encode("utf-8")).hexdigest()
    return os.path.join(index_dir, "terms", f"{h}.pkl")


def build_disk(st, folder: str, read_fn: Callable[[str], Tuple[str, str]] = read_txt):
    """Build disk-based index from folder (term-per-file organization)."""
    tmp = defaultdict(dict)  # term -> {doc_id: tf}
    
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            title, text = read_fn(p)
            did = st["next_id"]
            st["next_id"] += 1
            toks = tokenize(text, st.get("tokenizer"))
            st["doc_len"][did] = len(toks)
            st["titles"][did] = title
            for t, tf in Counter(toks).items():
                tmp[t][did] = tf
    
    # Write term files
    for term, post in tmp.items():
        path = _term_path(st["dir"], term)
        with open(path, "wb") as f:
            pickle.dump(post, f)
        st["lex"][term] = {
            "path": path,
            "df": len(post),
            "cf": int(sum(post.values()))
        }
    
    # Save lexicon and metadata
    with open(os.path.join(st["dir"], "lexicon.pkl"), "wb") as f:
        pickle.dump(st["lex"], f)
    
    with open(os.path.join(st["dir"], "meta.pkl"), "wb") as f:
        pickle.dump({"titles": st["titles"], "doc_len": st["doc_len"]}, f)


def load_disk_min(st):
    """Load only lexicon and metadata (minimal memory footprint)."""
    lx = os.path.join(st["dir"], "lexicon.pkl")
    if os.path.exists(lx):
        with open(lx, "rb") as f:
            st["lex"] = pickle.load(f)
    
    meta = os.path.join(st["dir"], "meta.pkl")
    if os.path.exists(meta):
        with open(meta, "rb") as f:
            m = pickle.load(f)
            st["titles"] = m.get("titles", {})
            st["doc_len"] = m.get("doc_len", {})
            if st["doc_len"]:
                st["next_id"] = max(st["doc_len"].keys()) + 1


def postings_disk(st, term: str) -> Dict[int, int]:
    """Get postings from disk (lazy loading - only relevant term file loaded)."""
    info = st["lex"].get(term)
    if not info:
        return {}
    with open(info["path"], "rb") as f:
        return pickle.load(f)


# =============================================================================
# (c) UPDATABLE (aux RAM + tombstone + merge to disk)
# Reference: IIR §4.5
# =============================================================================
def init_upd(index_dir="updindex", tokenizer=None):
    """Initialize updatable indexer (disk + auxiliary RAM index)."""
    base = init_disk(index_dir, tokenizer)
    load_disk_min(base)
    base.update({
        "aux": defaultdict(dict),  # term -> {doc_id: tf} (RAM)
        "deleted": set()  # tombstone pattern
    })
    return base


def add_upd(st, title: str, text: str):
    """Add new document to auxiliary RAM index."""
    did = st["next_id"]
    st["next_id"] += 1
    toks = tokenize(text, st.get("tokenizer"))
    st["doc_len"][did] = len(toks)
    st["titles"][did] = title
    for t, tf in Counter(toks).items():
        st["aux"][t][did] = tf
    return did


def delete_upd(st, doc_id: int):
    """Delete document using tombstone pattern."""
    st["deleted"].add(doc_id)


def update_upd(st, doc_id: int, title: str, text: str):
    """Update document: delete old + add new."""
    delete_upd(st, doc_id)
    return add_upd(st, title, text)


def postings_upd(st, term: str) -> Dict[int, int]:
    """Get postings from merged view (main disk + aux RAM - deleted)."""
    main = postings_disk(st, term) if st["lex"].get(term) else {}
    aux = st["aux"].get(term, {})
    merged = dict(main)
    merged.update(aux)
    
    if st["deleted"]:
        for d in list(merged.keys()):
            if d in st["deleted"]:
                del merged[d]
    
    return merged


def merge_upd(st):
    """Merge auxiliary index into main disk index."""
    # Merge terms in auxiliary index
    for term in list(st["aux"].keys()):
        main = postings_disk(st, term) if st["lex"].get(term) else {}
        main.update(st["aux"][term])
        
        if st["deleted"]:
            for d in list(main.keys()):
                if d in st["deleted"]:
                    del main[d]
        
        # Write / remove
        if main:
            path = _term_path(st["dir"], term)
            with open(path, "wb") as f:
                pickle.dump(main, f)
            st["lex"][term] = {
                "path": path,
                "df": len(main),
                "cf": int(sum(main.values()))
            }
        else:
            if term in st["lex"]:
                try:
                    os.remove(st["lex"][term]["path"])
                except OSError:
                    pass
                del st["lex"][term]
    
    # Update lexicon and metadata
    with open(os.path.join(st["dir"], "lexicon.pkl"), "wb") as f:
        pickle.dump(st["lex"], f)
    
    with open(os.path.join(st["dir"], "meta.pkl"), "wb") as f:
        pickle.dump({"titles": st["titles"], "doc_len": st["doc_len"]}, f)
    
    st["aux"].clear()
    st["deleted"].clear()
