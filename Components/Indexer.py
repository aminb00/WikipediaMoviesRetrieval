"""
Tiny IR Indexer (Functional, Minimal)
(a) Memory SPIMI (5 pts)  (b) Disk term-per-file (+2)  (c) Updatable aux+merge (+2)

References: Manning et al. IIR
- SPIMI: IIR §4.3
- Disk-based: IIR §5.2
- Compression (Gap + VB): IIR §5.3
- Dynamic Indexing: IIR §4.5
"""

import os
import pickle
import hashlib
from collections import defaultdict, Counter


# --- Common ---
def read_txt(path: str):
    """Read text file: returns (title, content)."""
    with open(path, "r", encoding="utf-8") as f:
        return os.path.basename(path), f.read()


# =============================================================================
# (a) MEMORY (RAM SPIMI)
# Reference: IIR §4.3
# =============================================================================
def init_memory(tokenize_fn):
    """Initialize memory-resident indexer state."""
    return {
        "index": defaultdict(dict),  # term -> {doc_id: tf}
        "doc_len": {},
        "titles": {},
        "next_id": 0,
        "tokenize": tokenize_fn
    }


def index_doc_mem(st, title: str, text: str):
    """Index a single document using SPIMI algorithm."""
    did = st["next_id"]
    st["next_id"] += 1
    toks = st["tokenize"](text)
    st["doc_len"][did] = len(toks)
    st["titles"][did] = title
    for t, tf in Counter(toks).items():
        st["index"][t][did] = tf
    return did


def build_memory(st, folder: str, read_fn=read_txt):
    """Build memory index from folder of documents."""
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            title, text = read_fn(p)
            index_doc_mem(st, title, text)


def postings_mem(st, term: str):
    """Get postings list for a term: {doc_id: tf}."""
    return dict(st["index"].get(term, {}))


# =============================================================================
# (b) DISK (term-per-file, only relevant parts loaded)
# Reference: IIR §5.2
# =============================================================================
def init_disk(index_dir="index", tokenize_fn=None):
    """Initialize disk-based indexer state."""
    os.makedirs(os.path.join(index_dir, "terms"), exist_ok=True)
    return {
        "dir": index_dir,
        "lex": {},  # term -> {"path": path, "df": int, "cf": int}
        "doc_len": {},
        "titles": {},
        "next_id": 0,
        "tokenize": tokenize_fn
    }


def _term_path(index_dir: str, term: str):
    """Generate file path for term's postings using MD5 hash."""
    h = hashlib.md5(term.encode("utf-8")).hexdigest()
    return os.path.join(index_dir, "terms", f"{h}.pkl")


# =============================================================================
# Compression: Gap Encoding + Variable-Byte (IIR §5.3)
# =============================================================================
def _gap_encode(postings: dict):
    """Convert doc_ids to gaps: [1,3,5] -> [1,2,2]. Reference: IIR §5.3."""
    if not postings:
        return []
    sorted_docs = sorted(postings.keys())
    gaps = []
    prev = 0
    for doc_id in sorted_docs:
        gaps.append((doc_id - prev, postings[doc_id]))
        prev = doc_id
    return gaps




def _vb_encode(n: int):
    """Variable-Byte encoding. Reference: IIR §5.3, Fig 5.8 (PREPEND order)."""
    bytes_list = []
    while True:
        bytes_list.insert(0, n % 128)  # PREPEND: most significant byte first
        if n < 128:
            break
        n = n // 128
    bytes_list[-1] += 128  # Last byte: continuation bit = 1
    return bytes(bytes_list)


def _vb_decode(byte_list):
    """Variable-Byte decoding. Reference: IIR §5.3, Fig 5.8 (VBDECODE algorithm)."""
    n = 0
    for byte_val in byte_list:
        n = 128 * n + (byte_val % 128)
        if byte_val >= 128:  # Continuation bit = 1 -> number complete
            break
    return n


def _compress_postings(postings: dict):
    """Compress postings: gap encoding + VB. Returns compressed bytes. Reference: IIR §5.3."""
    if not postings:
        return b""
    gaps = _gap_encode(postings)
    
    compressed = bytearray()
    for gap, tf in gaps:
        compressed.extend(_vb_encode(gap))
        compressed.extend(_vb_encode(tf))
    return bytes(compressed)


def _decompress_postings(data: bytes):
    """Decompress postings: VB decode + gap decode. Returns {doc_id: tf}. Reference: IIR §5.3."""
    if not data:
        return {}
    postings = {}
    i = 0
    doc_id = 0
    while i < len(data):
        # Decode gap
        gap_bytes = bytearray()
        while i < len(data) and data[i] < 128:
            gap_bytes.append(data[i])
            i += 1
        if i < len(data):
            gap_bytes.append(data[i])
            i += 1
        gap = _vb_decode(gap_bytes)
        doc_id += gap
        
        # Decode tf
        tf_bytes = bytearray()
        while i < len(data) and data[i] < 128:
            tf_bytes.append(data[i])
            i += 1
        if i < len(data):
            tf_bytes.append(data[i])
            i += 1
        tf = _vb_decode(tf_bytes)
        postings[doc_id] = tf
    return postings


def build_disk(st, folder: str, read_fn=read_txt):
    """Build disk-based index from folder (term-per-file organization)."""
    tmp = defaultdict(dict)  # term -> {doc_id: tf}
    
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            title, text = read_fn(p)
            did = st["next_id"]
            st["next_id"] += 1
            toks = st["tokenize"](text)
            st["doc_len"][did] = len(toks)
            st["titles"][did] = title
            for t, tf in Counter(toks).items():
                tmp[t][did] = tf
    
    # Write term files (compressed: gap + VB encoding, IIR §5.3)
    for term, post in tmp.items():
        path = _term_path(st["dir"], term)
        compressed = _compress_postings(post)
        with open(path, "wb") as f:
            f.write(compressed)
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


def postings_disk(st, term: str):
    """Get postings from disk (lazy loading - only relevant term file loaded).
    Uses gap + VB compression (IIR §5.3)."""
    info = st["lex"].get(term)
    if not info:
        return {}
    with open(info["path"], "rb") as f:
        data = f.read()
    return _decompress_postings(data)


# =============================================================================
# (c) UPDATABLE (aux RAM + tombstone + merge to disk)
# Reference: IIR §4.5
# =============================================================================
def init_upd(index_dir="updindex", tokenize_fn=None, merge_threshold=100):
    """Initialize updatable indexer (disk + auxiliary RAM index).
    
    Args:
        merge_threshold: Auto-merge when auxiliary has this many documents (IIR §4.5)
    """
    base = init_disk(index_dir, tokenize_fn)
    load_disk_min(base)
    base.update({
        "aux": defaultdict(dict),  # term -> {doc_id: tf} (RAM)
        "deleted": set(),  # tombstone pattern
        "merge_threshold": merge_threshold
    })
    return base


def add_upd(st, title: str, text: str):
    """Add new document to auxiliary RAM index.
    Auto-merges if threshold exceeded (IIR §4.5)."""
    did = st["next_id"]
    st["next_id"] += 1
    toks = st["tokenize"](text)
    st["doc_len"][did] = len(toks)
    st["titles"][did] = title
    for t, tf in Counter(toks).items():
        st["aux"][t][did] = tf
    
    # Auto-merge if threshold exceeded (IIR §4.5: batch merge for efficiency)
    aux_doc_count = len(set(d for term_postings in st["aux"].values() for d in term_postings.keys()))
    if aux_doc_count >= st.get("merge_threshold", 100):
        merge_upd(st)
    
    return did


def delete_upd(st, doc_id: int):
    """Delete document using tombstone pattern."""
    st["deleted"].add(doc_id)


def update_upd(st, doc_id: int, title: str, text: str):
    """Update document: delete old + add new."""
    delete_upd(st, doc_id)
    return add_upd(st, title, text)


def postings_upd(st, term: str):
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
    # First, collect all terms that need updating (aux terms + all existing terms for deletions)
    terms_to_process = set(st["aux"].keys())
    
    # If there are deletions, we need to check ALL existing terms
    if st["deleted"]:
        terms_to_process.update(st["lex"].keys())
    
    # Process each term
    for term in terms_to_process:
        # Get main postings from disk if exists
        main = postings_disk(st, term) if st["lex"].get(term) else {}
        
        # Update with aux if present
        if term in st["aux"]:
            main.update(st["aux"][term])
        
        # Remove deleted documents
        if st["deleted"]:
            for d in list(main.keys()):
                if d in st["deleted"]:
                    del main[d]
        
        # Write / remove (compressed: gap + VB, IIR §5.3)
        if main:
            path = _term_path(st["dir"], term)
            compressed = _compress_postings(main)
            with open(path, "wb") as f:
                f.write(compressed)
            st["lex"][term] = {
                "path": path,
                "df": len(main),
                "cf": int(sum(main.values()))
            }
        else:
            # Empty postings - remove term file and lexicon entry
            if term in st["lex"]:
                try:
                    os.remove(st["lex"][term]["path"])
                except OSError:
                    pass
                del st["lex"][term]
    
    # Clean up deleted documents from metadata
    if st["deleted"]:
        for d in st["deleted"]:
            if d in st["titles"]:
                del st["titles"][d]
            if d in st["doc_len"]:
                del st["doc_len"][d]
    
    # Update lexicon and metadata
    with open(os.path.join(st["dir"], "lexicon.pkl"), "wb") as f:
        pickle.dump(st["lex"], f)
    
    with open(os.path.join(st["dir"], "meta.pkl"), "wb") as f:
        pickle.dump({"titles": st["titles"], "doc_len": st["doc_len"]}, f)
    
    # Preserve merge_threshold in state
    threshold = st.get("merge_threshold", 100)
    st["aux"].clear()
    st["deleted"].clear()
    st["merge_threshold"] = threshold
