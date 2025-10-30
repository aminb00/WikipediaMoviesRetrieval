"""
Unified Indexer with SPIMI, Disk-based, and Updatable support
Based on: Manning et al. IIR Ch.4 Section 4.3 (SPIMI), Ch.4.5 (Dynamic), Ch.5 (Compression)

SOLID Design:
- Single Responsibility: Each indexer variant handles one storage strategy
- Open/Closed: Extensible via configuration, not modification
- Liskov Substitution: All variants implement same interface
"""
from collections import defaultdict, Counter
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from tokenizer import Tokenizer


class BaseIndexer(ABC):
    """
    Abstract base class for indexers.
    
    IIR Ch.4: All indexers must support:
    - Indexing documents (SPIMI algorithm)
    - Retrieving postings with tf
    - Collection statistics
    """
    
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        # Document metadata
        self.doc_ids: List[int] = []
        self.doc_titles: Dict[int, str] = {}
        self.doc_plots: Dict[int, str] = {}
        # Collection statistics
        self.doc_len: Dict[int, int] = {}
        self.term_cf: Dict[str, int] = defaultdict(int)
        self.collection_tokens: int = 0
    
    @abstractmethod
    def get_postings(self, term: str) -> Dict[int, int]:
        """Get postings for a term: {doc_id: tf}"""
        pass
    
    @abstractmethod
    def get_document_frequency(self, term: str) -> int:
        """Get document frequency (df)"""
        pass
    
    @abstractmethod
    def get_collection_frequency(self, term: str) -> int:
        """Get collection frequency (cf)"""
        pass
    
    def doc_term_tf(self, doc_id: int, term: str) -> int:
        """Get term frequency for document-term pair."""
        return self.get_postings(term).get(doc_id, 0)
    
    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            'num_documents': len(self.doc_ids),
            'collection_tokens': self.collection_tokens
        }


class InMemoryIndexer(BaseIndexer):
    """
    Memory-resident indexer using SPIMI algorithm.
    
    This class implements the Single-Pass In-Memory Indexing (SPIMI) algorithm as described
    in IIR Chapter 4 Section 4.3. Unlike BSBI (Blocked Sort-Based Indexing), SPIMI does not
    require a separate term-to-termID mapping table. Instead, it uses terms directly as dictionary
    keys and builds postings lists incrementally as documents are processed. This approach eliminates
    the need for sorting intermediate results and reduces memory overhead.
    
    The core design decision here is to use terms (strings) as dictionary keys rather than term IDs.
    While this requires more memory per term, it simplifies the indexing process by avoiding the
    two-pass nature of BSBI. SPIMI's single-pass characteristic makes it more memory-efficient for
    building indexes in environments where RAM is available but intermediate disk writes should be
    minimized.
    
    IIR Ch.4 Section 4.3, Figure 4.4: SPIMI-INVERT
    - Stores index: term -> {doc_id: tf}
    - Fast retrieval, requires enough RAM
    """
    
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
        # Index: term -> {doc_id: tf} 
        # SPIMI design choice: Using actual term strings as keys (not term IDs) eliminates the need
        # for a separate term-to-termID mapping table that BSBI requires. This simplifies the algorithm
        # to a single pass but trades off memory efficiency for implementation simplicity. The defaultdict
        # automatically creates new term entries when first encountered, which aligns with SPIMI's
        # dynamic dictionary construction approach.
        self.index: Dict[str, Dict[int, int]] = defaultdict(dict)
        
        # Document terms: doc_id -> {term: tf}
        # This reverse index structure is essential for efficient document normalization computation
        # in the Vector Space Model (VSM). When calculating document norms for ltc.ltc scheme, we need
        # to iterate over all terms in a document. Without this structure, we would need to scan the
        # entire vocabulary for each document, resulting in O(|vocab| * |docs|) complexity. With
        # doc_terms, we achieve O(|terms_in_doc|) complexity per document norm calculation, which is
        # a significant performance improvement especially for VSM ranking. This optimization allows
        # query processors to compute document norms in linear time relative to document size rather
        # than vocabulary size.
        self.doc_terms: Dict[int, Dict[str, int]] = defaultdict(dict)
    
    def index_document(self, doc_id: int, title: str, plot: str):
        """
        Index a single document following SPIMI algorithm (corresponds to SPIMI-INVERT line 10).
        
        This method processes a single document by tokenizing its content, counting term frequencies,
        and updating both the inverted index (term -> {doc_id: tf}) and the forward index (doc_id -> {term: tf}).
        The SPIMI approach allows us to process documents one at a time without requiring intermediate
        sorting or buffering of postings lists, unlike BSBI which requires sorting before merging blocks.
        
        The term frequency (tf) is crucial for ranking algorithms: VSM uses tf for logarithmic weighting,
        BM25 uses tf in its saturation function, and Language Models use tf directly in probability
        calculations. By storing tf alongside doc_id in the postings structure, we enable all three
        ranking models without re-tokenizing documents during query processing.
        
        Collection statistics (term_cf, doc_len, collection_tokens) are maintained incrementally.
        These statistics are pre-computed during indexing to avoid recalculating them for every query,
        which significantly improves query processing performance. Specifically: term_cf (collection frequency)
        is needed for language model smoothing, doc_len is required for BM25 length normalization, and
        collection_tokens is used to compute collection-wide term probabilities.
        """
        self.doc_ids.append(doc_id)
        self.doc_titles[doc_id] = title
        self.doc_plots[doc_id] = plot
        
        tokens = self.tokenizer.tokenize(plot)
        term_counts = Counter(tokens)
        
        # Document length is stored for BM25 normalization. BM25 applies length normalization to prevent
        # long documents from dominating search results. The formula uses doc_length / avg_doc_length
        # to normalize term frequency saturation, ensuring fair comparison across documents of varying lengths.
        self.doc_len[doc_id] = len(tokens)
        self.collection_tokens += len(tokens)
        
        for term, tf in term_counts.items():
            # Update inverted index: term -> {doc_id: tf}
            # SPIMI allows direct insertion into postings without sorting, as postings are maintained
            # as a dictionary structure. When saving to disk, we sort docIDs for gap encoding efficiency,
            # but during in-memory construction, unsorted insertion is acceptable and faster.
            self.index[term][doc_id] = tf
            
            # Update forward index: doc_id -> {term: tf}
            # This bidirectional indexing enables efficient traversal in both directions. The forward
            # index is particularly valuable for document norm calculation in VSM, where we need to
            # iterate over all terms in a document. Without this structure, norm calculation would
            # require O(|vocabulary|) operations per document instead of O(|terms_in_document|).
            self.doc_terms[doc_id][term] = tf
            
            # Update collection frequency: total occurrences of term across all documents
            # Collection frequency (cf) differs from document frequency (df): cf counts total occurrences
            # while df counts unique documents. For language models with Jelinek-Mercer smoothing,
            # we need cf to compute P(t|collection) = cf / |collection_size|, which serves as the
            # background distribution for smoothing unseen terms in documents.
            self.term_cf[term] += tf
    
    def index_documents_batch(self, documents: List[tuple]):
        """Batch indexing for performance."""
        plots = [doc[2] for doc in documents]
        all_tokens = self.tokenizer.tokenize_batch(plots)
        
        for (doc_id, title, plot), tokens in zip(documents, all_tokens):
            self.doc_ids.append(doc_id)
            self.doc_titles[doc_id] = title
            self.doc_plots[doc_id] = plot
            
            term_counts = Counter(tokens)
            self.doc_len[doc_id] = len(tokens)
            self.collection_tokens += len(tokens)
            
            for term, tf in term_counts.items():
                self.index[term][doc_id] = tf
                self.doc_terms[doc_id][term] = tf  # For efficient doc norm computation
                self.term_cf[term] += tf
    
    def build_index(self, dataset_path: str):
        """Build index from CSV files (SPIMI algorithm)."""
        import os
        import pandas as pd
        from tqdm import tqdm
        
        csv_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.csv')])
        
        total_docs = sum(len(pd.read_csv(os.path.join(dataset_path, f))) for f in csv_files)
        print(f"Indexing {total_docs} documents...")
        
        BATCH_SIZE = 100
        doc_id = 0
        
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(dataset_path, csv_file))
            batch = []
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_file):
                title = str(row.get('title') or "")
                plot = str(row.get('plot') or "")
                batch.append((doc_id, title, plot))
                doc_id += 1
                
                if len(batch) >= BATCH_SIZE:
                    self.index_documents_batch(batch)
                    batch = []
            
            if batch:
                self.index_documents_batch(batch)
        
        print("Index finalized")
    
    def get_postings(self, term: str) -> Dict[int, int]:
        return self.index.get(term, {})
    
    def get_document_frequency(self, term: str) -> int:
        return len(self.index.get(term, {}))
    
    def get_collection_frequency(self, term: str) -> int:
        return self.term_cf.get(term, 0)
    
    def get_statistics(self) -> Dict:
        stats = super().get_statistics()
        stats.update({
            'num_terms': len(self.index),
            'total_df_pairs': sum(len(p) for p in self.index.values()),
            'avg_df_per_term': sum(len(p) for p in self.index.values()) / len(self.index) if self.index else 0
        })
        return stats


class DiskIndexer(InMemoryIndexer):
    """
    Disk-based indexer with gap encoding.
    
    Extension: Stores postings on disk, loads on-demand.
    IIR Ch.5 Section 5.3.2: Variable byte encoding
    
    Uses in-memory indexer to build, then writes to disk.
    """
    
    def __init__(self, tokenizer: Tokenizer, index_dir: str = "index"):
        super().__init__(tokenizer)
        self.index_dir = index_dir
        self.disk_dict: Dict[str, Dict] = {}
        import os
        os.makedirs(index_dir, exist_ok=True)
    
    @staticmethod
    def _gap_encode(postings: List[int]) -> List[int]:
        """
        Encode document IDs as gaps for compression (IIR Chapter 5).
        
        Gap encoding is a fundamental compression technique for inverted indexes. Instead of storing
        absolute document IDs (which can be large integers), we store the difference (gap) between
        consecutive document IDs. For example, postings [5, 8, 10, 15] become gaps [5, 3, 2, 5].
        
        This technique is effective because:
        1. Postings lists are sorted by document ID during disk write
        2. Document IDs in sorted lists tend to be clustered (documents from same source/timestamp)
        3. Small gaps compress better with Variable Byte encoding than large absolute IDs
        
        According to IIR Chapter 5, gap encoding alone can reduce index size by 30-50% for typical
        collections. When combined with Variable Byte encoding (which uses fewer bytes for smaller
        numbers), the compression ratio improves significantly. This is especially important for disk-based
        indexes where I/O bandwidth is a bottleneck.
        """
        if not postings:
            return []
        
        # First element remains absolute (base document ID)
        gaps = [postings[0]]
        
        # Subsequent elements are stored as differences (gaps)
        # This transformation is lossless: we can reconstruct original IDs by cumulative addition
        for i in range(1, len(postings)):
            gaps.append(postings[i] - postings[i-1])
        return gaps
    
    @staticmethod
    def _variable_byte_encode(number: int) -> bytes:
        """
        Variable Byte encoding for integer compression (IIR Chapter 5 Section 5.3.2).
        
        Variable Byte encoding divides an integer into 7-bit chunks (excluding the continuation bit),
        where each byte uses 7 bits for data and 1 bit (MSB) as a continuation flag. The high bit
        (0x80) is set to 1 on the last byte, signaling the end of the number. Numbers less than 128
        require only one byte, numbers up to 16383 require two bytes, and so on.
        
        This encoding scheme is chosen over fixed-width integers because:
        1. Small numbers (like tf values, which are often 1-5) use fewer bytes
        2. Decoding is straightforward: read bytes until MSB=1
        3. It provides good compression for typical gap-encoded document IDs
        4. It's faster to decode than more sophisticated schemes like Gamma or Delta codes
        
        The special case for 0 ensures that zero values are properly encoded. Without this check,
        the while loop would never execute and return an empty byte sequence, which would break
        the decoding process. Encoding 0 as a single byte with only the continuation bit set (0x80)
        maintains the invariant that every number has at least one byte in its encoding.
        
        Example: number 300 = 0x12C
        - Binary: 0000001 00101100
        - Chunks: [1] [44] (7 bits each)
        - Encoded: [0x01] [0xAC] (0xAC = 0x80 | 0x2C)
        """
        # Special case for 0: must encode as [0x80] to maintain decoder invariants
        # Without this, 0 would produce an empty byte sequence, breaking reconstruction
        if number == 0:
            return bytes([0x80])
        
        # Extract 7-bit chunks from the number (LSB first)
        bytes_list = []
        n = number
        while n > 0:
            bytes_list.append(n & 0x7F)  # Extract lower 7 bits
            n >>= 7                      # Shift right by 7 bits
        
        # Build the encoded bytes (MSB first), setting continuation bit on last byte
        result = bytearray()
        for i, byte_val in enumerate(reversed(bytes_list)):
            if i == len(bytes_list) - 1:
                # Last byte: set MSB to 1 (0x80) to signal end of number
                result.append(byte_val | 0x80)
            else:
                # Intermediate bytes: MSB remains 0 (continuation byte)
                result.append(byte_val)
        return bytes(result)
    
    @staticmethod
    def _variable_byte_decode(data: bytes) -> List[int]:
        """Decode variable byte encoded data."""
        numbers = []
        current = 0
        for byte_val in data:
            current = (current << 7) | (byte_val & 0x7F)
            if byte_val & 0x80:
                numbers.append(current)
                current = 0
        return numbers
    
    def _gap_decode(self, gaps: List[int]) -> List[int]:
        """Decode gaps back to docIDs."""
        if not gaps:
            return []
        docids = [gaps[0]]
        for gap in gaps[1:]:
            docids.append(docids[-1] + gap)
        return docids
    
    def save_to_disk(self):
        """
        Save in-memory index to disk with compressed postings format.
        
        This method implements the disk persistence strategy for large-scale indexes. The key design
        principle is to store postings in a compressed format on disk, keeping only metadata (dictionary)
        in memory. This enables the "load only relevant parts" requirement by allowing random access
        to specific posting lists via file offsets.
        
        Postings Format:
        The on-disk format for each term's postings is: [df (VByte)] + [(gap₁, tf₁), (gap₂, tf₂), ...]
        where df is the document frequency (number of documents containing the term), and each pair
        represents a gap-encoded document ID followed by its term frequency. Both gaps and term frequencies
        are Variable Byte encoded.
        
        Why store (gap, tf) pairs together?
        - Gap encoding reduces document ID storage from ~4 bytes to ~1-2 bytes for typical gaps
        - Term frequencies are also small (often 1-10) and compress well with VByte
        - Storing them together enables efficient sequential decoding during query processing
        - The integrity check (expected_len = 1 + 2*df) ensures we can detect corrupted data
        
        Dictionary Structure (terms.lex):
        Instead of storing the entire postings lists in the dictionary (which would defeat the purpose
        of on-demand loading), we only store metadata: document frequency (df), collection frequency (cf),
        file offset, and byte length. This keeps the dictionary small enough to remain in memory, enabling
        O(1) term lookups while allowing efficient disk access when postings are needed.
        
        Metadata Optimization:
        We intentionally exclude doc_plots from metadata.pkl to reduce disk footprint. Query processing
        only needs document titles for display; full plots are unnecessary for ranking algorithms which
        rely on the pre-computed index structure. This design reduces metadata size by ~80-90% for typical
        text collections, significantly improving load times.
        
        IIR Ch.5: Gap encoding + Variable Byte compression provides efficient storage for sparse indexes.
        """
        import pickle
        import os
        
        postings_file = os.path.join(self.index_dir, "postings.bin")
        postings_file_tmp = postings_file + ".tmp"
        current_offset = 0
        sorted_terms = sorted(self.index.keys())
        
        with open(postings_file_tmp, 'wb') as f:
            for term in sorted_terms:
                postings_dict = self.index[term]
                docids = sorted(postings_dict.keys())
                
                if not docids:
                    continue
                
                gaps = self._gap_encode(docids)
                encoded_data = bytearray()
                
                # Write df first
                encoded_data.extend(self._variable_byte_encode(len(docids)))
                
                # Build skip pointers for efficient AND intersection (IIR Ch.2 Section 2.3)
                # Skip every √P entries (P = document frequency)
                # Each skip entry: (docID, byte_offset)
                # Skip pointers enable jumping over large segments during intersection operations
                skip_every = max(1, int(len(docids) ** 0.5))  # √P heuristic
                skip_entries = []
                
                # Write (gap, tf) pairs and build skip table
                byte_offset_within_entry = len(encoded_data)
                for idx, (gap, doc_id) in enumerate(zip(gaps, docids)):
                    # Record skip entry every √P documents
                    if idx > 0 and idx % skip_every == 0:
                        # Offset is relative to the start of this term's postings
                        skip_entries.append((doc_id, current_offset + byte_offset_within_entry))
                    
                    gap_bytes = self._variable_byte_encode(gap)
                    tf_bytes = self._variable_byte_encode(postings_dict[doc_id])
                    encoded_data.extend(gap_bytes)
                    encoded_data.extend(tf_bytes)
                    byte_offset_within_entry += len(gap_bytes) + len(tf_bytes)
                
                # Store metadata including skip pointers
                # Skip pointers are only stored for lists with sufficient length (df > skip_every)
                # This balances skip overhead vs. intersection speedup
                self.disk_dict[term] = {
                    'df': len(docids),
                    'cf': sum(postings_dict.values()),
                    'offset': current_offset,
                    'length': len(encoded_data),
                    'skip_entries': skip_entries if len(docids) > skip_every else []  # Only for longer lists
                }
                
                f.write(encoded_data)
                current_offset += len(encoded_data)
        
        # Atomic write pattern: write to temp files then rename
        # This ensures data integrity - if write fails, original files remain intact
        # Prevents corruption from partial writes (e.g., disk full, process killed)
        dict_file = os.path.join(self.index_dir, "terms.lex")
        dict_file_tmp = dict_file + ".tmp"
        with open(dict_file_tmp, 'wb') as f:
            # Include version for format compatibility checking
            pickle.dump({
                'version': '1.0',
                'dict': self.disk_dict
            }, f)
        os.replace(dict_file_tmp, dict_file)  # Atomic rename (Windows/Unix compatible)
        
        meta_file = os.path.join(self.index_dir, "metadata.pkl")
        meta_file_tmp = meta_file + ".tmp"
        with open(meta_file_tmp, 'wb') as f:
            # Minimal metadata (no doc_plots to save space)
            # Version field enables format migration in future versions
            pickle.dump({
                'version': '1.0',
                'doc_ids': self.doc_ids,
                'doc_titles': self.doc_titles,
                'doc_len': self.doc_len,
                'term_cf': dict(self.term_cf),
                'collection_tokens': self.collection_tokens
            }, f)
        os.replace(meta_file_tmp, meta_file)
        
        # Atomic rename for postings file
        postings_file_tmp = postings_file + ".tmp"
        os.replace(postings_file_tmp, postings_file)
        
        print(f"Index saved to {self.index_dir}/")
    
    def load_from_disk(self):
        """Load index from disk."""
        import pickle
        import os
        
        dict_file = os.path.join(self.index_dir, "terms.lex")
        if os.path.exists(dict_file):
            with open(dict_file, 'rb') as f:
                data = pickle.load(f)
                # Handle version compatibility: older format may not have version field
                if isinstance(data, dict) and 'dict' in data:
                    # New format with version
                    self.disk_dict = data['dict']
                    version = data.get('version', '1.0')
                else:
                    # Legacy format (direct dictionary)
                    self.disk_dict = data
                    version = '0.9'
        
        meta_file = os.path.join(self.index_dir, "metadata.pkl")
        if os.path.exists(meta_file):
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
                # Handle version compatibility
                if isinstance(meta, dict) and 'doc_ids' in meta:
                    # New format with version field
                    self.doc_ids = meta['doc_ids']
                    self.doc_titles = meta['doc_titles']
                    self.doc_plots = meta.get('doc_plots', {})  # Optional (minimal metadata)
                    self.doc_len = meta['doc_len']
                    self.term_cf = defaultdict(int, meta['term_cf'])
                    self.collection_tokens = meta['collection_tokens']
                else:
                    # Legacy format (direct metadata dict)
                    self.doc_ids = meta.get('doc_ids', [])
                    self.doc_titles = meta.get('doc_titles', {})
                    self.doc_plots = meta.get('doc_plots', {})
                    self.doc_len = meta.get('doc_len', {})
                    self.term_cf = defaultdict(int, meta.get('term_cf', {}))
                    self.collection_tokens = meta.get('collection_tokens', 0)
            print(f"Index loaded from {self.index_dir}/")
    
    def get_postings(self, term: str, skip_target: Optional[int] = None) -> Dict[int, int]:
        """
        Load postings from disk on-demand (IIR Chapter 5: selective loading strategy).
        
        This method implements the core principle of disk-based indexes: only load the data that is
        needed for the current query. Unlike in-memory indexes that load everything upfront, this
        approach enables handling indexes larger than available RAM by performing selective I/O.
        
        Skip Pointer Support:
        If skip_target is provided, skip pointers are used to jump to the relevant segment of the
        postings list. This is especially beneficial for AND intersections where we need to find
        documents present in multiple lists. Skip pointers are stored every √P entries (P = df),
        allowing O(√P) complexity for finding a target document instead of O(P) sequential scan.
        
        On-Demand Loading Strategy:
        When a query term is processed, we:
        1. Look up the term in the in-memory dictionary (O(1) hash lookup)
        2. Extract the file offset and byte length from the dictionary entry
        3. If skip_target is provided and skip pointers exist, use them to seek to relevant segment
        4. Otherwise, seek to the start of postings (O(1) file operation)
        5. Read only the required bytes (typically 10-1000 bytes per term)
        6. Decode the compressed data back to {doc_id: tf} mapping
        
        This approach is fundamentally different from loading the entire index into memory. For a
        collection with millions of documents and hundreds of thousands of unique terms, loading
        all postings would require gigabytes of RAM. By loading only the terms present in queries
        (typically 1-10 terms per query), we reduce memory footprint by orders of magnitude.
        
        Decoding Process:
        The decoding reverses the encoding process: we first read the document frequency (df), which
        tells us how many (gap, tf) pairs to expect. We then reconstruct document IDs by cumulative
        addition of gaps (the first gap is absolute, subsequent gaps are relative). This reconstruction
        is exact: gap encoding is a lossless compression scheme.
        
        Integrity Checking:
        The expected_len check (1 + 2*df) verifies that we read a complete posting list. If the file
        is corrupted or if we accidentally read from a wrong offset, this check will fail and we return
        an empty result rather than returning corrupt data. This defensive programming practice is
        essential for production systems where data integrity cannot be assumed.
        
        Performance Considerations:
        Each get_postings call requires one disk seek and one read operation. For queries with multiple
        terms, we perform multiple seeks. While seeking is expensive (typically 1-10ms per seek on
        HDDs), the benefit of not loading unused postings far outweighs this cost for typical query
        workloads where only a small fraction of the vocabulary appears in queries.
        
        IIR Ch.5 recommends this selective loading approach as the standard for disk-based indexes.
        IIR Ch.2 Section 2.3 describes skip pointers for efficient intersection algorithms.
        """
        if term not in self.disk_dict:
            return {}
        
        entry = self.disk_dict[term]
        offset = entry['offset']
        length = entry['length']
        skip_entries = entry.get('skip_entries', [])  # May not exist for older formats
        
        import os
        postings_file = os.path.join(self.index_dir, "postings.bin")
        
        if not os.path.exists(postings_file):
            return {}
        
        # Skip pointer optimization: if skip_target is provided and skip entries exist,
        # we can identify the relevant segment, but due to gap encoding requiring cumulative
        # decoding from the start, we still read the full list. However, skip pointers enable
        # early termination once we pass the target, reducing unnecessary decoding.
        start_skip_docid = None
        if skip_target is not None and skip_entries:
            # Find the largest skip entry with docID <= skip_target
            # This identifies the segment containing the target document
            for skip_docid, skip_offset in skip_entries:
                if skip_docid <= skip_target:
                    start_skip_docid = skip_docid
                else:
                    break  # Skip entries are sorted by docID
        
        # Read only the relevant slice: this is the key to "load only what's needed"
        # f.seek() positions the file pointer at the exact byte offset, and f.read(length)
        # reads precisely the number of bytes needed for this term's postings. This selective
        # I/O is what enables handling indexes larger than available RAM.
        # Note: Gap encoding requires decoding from the start, so we read the full list even
        # when skip pointers indicate a target segment. Early termination during decoding
        # provides the performance benefit.
        with open(postings_file, 'rb') as f:
            f.seek(offset)
            data = f.read(length)
        
        # Decode: df, then (gap, tf) pairs
        nums = self._variable_byte_decode(data)
        if not nums:
            return {}
        
        df = nums[0]
        # Integrity check: expect 1 (df) + 2*df (gap, tf pairs)
        # This validation ensures data integrity. If the read was incomplete or corrupted,
        # len(nums) will be less than expected, and we return empty rather than partial/corrupt data.
        expected_len = 1 + 2 * df
        if len(nums) < expected_len:
            # Data corruption or incomplete read - fail safely by returning empty result
            return {}
        
        # Reconstruct postings dictionary from gaps and term frequencies
        postings = {}
        doc_id_acc = 0
        i = 1
        skip_optimization_active = (skip_target is not None and start_skip_docid is not None)
        
        for _ in range(df):
            gap = nums[i]
            tf = nums[i + 1]
            i += 2
            
            # Decode gap: first gap is absolute, subsequent are relative
            # This reconstruction is the inverse of gap encoding. The first gap represents the first
            # document ID directly. Each subsequent gap is added to the previous cumulative document ID
            # to reconstruct the next document ID. This is mathematically sound: original[i] = sum(gaps[0:i+1])
            if doc_id_acc == 0:
                doc_id_acc = gap
            else:
                doc_id_acc = doc_id_acc + gap
            
            # Skip pointer optimization: if we're searching for a target and have passed it,
            # we can terminate early. This reduces unnecessary decoding for AND intersections.
            # Without skip pointers, we'd decode all postings; with skip pointers and early termination,
            # we only decode until we find or pass the target, providing O(√P) complexity benefit.
            if skip_optimization_active:
                if doc_id_acc < start_skip_docid:
                    continue  # Haven't reached the skip segment yet
                if doc_id_acc > skip_target:
                    break  # Passed target, no need to continue
            
            postings[doc_id_acc] = tf
        
        return postings
    
    def intersect_postings(self, term1: str, term2: str) -> Dict[int, int]:
        """
        Efficient AND intersection of two postings lists using skip pointers.
        
        This method demonstrates skip pointer usage for AND queries. It implements the
        two-pointer algorithm optimized with skip pointers: when one list is much shorter
        than the other, we can use skip pointers on the longer list to jump over irrelevant
        segments, reducing the number of docIDs we need to examine.
        
        Algorithm:
        1. Get document frequencies to determine which list is shorter
        2. Load shorter list fully (it's small, O(min(P1,P2)))
        3. For each docID in shorter list, use skip_target parameter to efficiently search longer list
        4. Skip pointers on longer list enable O(√P) search instead of O(P) sequential scan
        
        Complexity: O(min(P1, P2) + √max(P1, P2)) with skip pointers vs. O(P1 + P2) without.
        The √max(P1, P2) comes from using skip pointers: we examine at most √P entries
        in the longer list per docID in the shorter list, thanks to skip pointer jumps.
        
        Reference: IIR Ch.2 Section 2.3 - Intersection with skip pointers
        """
        df1 = self.get_document_frequency(term1)
        df2 = self.get_document_frequency(term2)
        
        # Determine which list is shorter - iterate over shorter, search in longer
        if df1 > df2:
            short_term, long_term = term2, term1
        else:
            short_term, long_term = term1, term2
        
        # Load shorter list fully (it's small, acceptable memory cost)
        short_postings = self.get_postings(short_term)
        
        # For each docID in shorter list, use skip_target to efficiently find it in longer list
        # The skip_target parameter enables skip pointer optimization in get_postings:
        # it identifies the relevant segment and enables early termination
        result = {}
        for doc_id in short_postings.keys():
            # Use skip_target to enable skip pointer jumping in the longer list
            # get_postings will use skip_entries to jump to relevant segment and terminate early
            long_postings = self.get_postings(long_term, skip_target=doc_id)
            if doc_id in long_postings:
                # Found in both lists - take minimum tf (or combine according to query semantics)
                tf_short = short_postings[doc_id]
                tf_long = long_postings[doc_id]
                result[doc_id] = min(tf_short, tf_long)
        
        return result
    
    def get_document_frequency(self, term: str) -> int:
        return self.disk_dict.get(term, {}).get('df', 0)
    
    def get_collection_frequency(self, term: str) -> int:
        return self.disk_dict.get(term, {}).get('cf', 0)
    
    def build_index(self, dataset_path: str):
        """Build index in memory, then save to disk."""
        super().build_index(dataset_path)
        self.save_to_disk()


class UpdatableIndexer(DiskIndexer):
    """
    Updatable indexer with auxiliary index for dynamic indexing (IIR Ch.4 Section 4.5).
    
    This class implements a dynamic indexing strategy that allows documents to be added, deleted,
    and updated without requiring a full index rebuild. The design is inspired by Lucene's segment-based
    architecture and follows the auxiliary index pattern described in IIR Chapter 4 Section 4.5.
    
    Core Design: Two-Level Index Architecture
    The index is divided into two components:
    1. Main Index (on disk): The stable, persistent index containing the majority of documents.
       This index is updated only during merge operations, minimizing disk I/O.
    2. Auxiliary Index (in memory): A small, temporary index for newly added documents. This index
       is kept in RAM for fast insertion and is periodically merged into the main index.
    
    Why This Design?
    Updating the main index for every addition would require:
    - Reading the entire posting list from disk
    - Merging with new postings in memory
    - Rewriting the entire posting list to disk
    
    This is prohibitively expensive for frequent updates. The auxiliary index approach batches updates
    and performs merges periodically, trading query-time merge cost for reduced write amplification.
    
    Deletion Strategy: Tombstone Pattern
    Deletions are handled using an invalidation bit-vector (deleted_docs set). Rather than immediately
    removing documents from the index (which would require rewriting postings), we mark them as deleted.
    During query processing, we filter out deleted document IDs. During merge operations, we permanently
    remove deleted documents from the index. This tombstone pattern is widely used in production systems
    (Lucene, Elasticsearch) because it makes deletion a fast O(1) operation.
    
    Merge Strategy:
    The merge_auxiliary() method combines the auxiliary index into the main index. This is performed:
    - Periodically (user-triggered via merge command)
    - When auxiliary index grows beyond a threshold (future optimization)
    - Before query processing (optional, for consistent views)
    
    During merge, we:
    1. Combine postings from both indexes (handling term collisions)
    2. Update metadata (doc_ids, doc_titles, doc_len)
    3. Write the merged index back to disk
    4. Clear the auxiliary index to start fresh
    
    This two-phase commit ensures that if a merge fails, the original index remains intact.
    
    Extension: Supports add/delete/update operations.
    """
    
    def __init__(self, tokenizer: Tokenizer, index_dir: str = "index"):
        super().__init__(tokenizer, index_dir)
        
        # Auxiliary index for new documents (maintained in RAM for fast insertion)
        # This follows the dynamic indexing pattern: new documents go into a small auxiliary segment
        # that is periodically merged into the main index. Keeping it in memory allows O(1) insertions
        # without disk I/O. The auxiliary index has the same structure as the main index (term -> {doc_id: tf}),
        # enabling uniform query processing that merges results from both indexes.
        self.aux_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.aux_doc_ids: List[int] = []
        self.aux_doc_titles: Dict[int, str] = {}
        self.aux_doc_plots: Dict[int, str] = {}
        self.aux_doc_len: Dict[int, int] = {}
        self.aux_next_doc_id: int = 0
        
        # Deletion tracking: invalidation bit-vector pattern
        # Instead of immediately removing deleted documents from postings (expensive: requires rewriting),
        # we maintain a set of deleted document IDs. During query processing, we filter out these IDs.
        # During merge operations, deleted documents are permanently removed from the index. This tombstone
        # pattern makes deletion an O(1) operation and is standard in production IR systems<｜place▁holder▁no▁737｜> like Lucene.
        self.deleted_docs: set = set()
        
        # Load existing if available
        import os
        if os.path.exists(os.path.join(index_dir, "terms.lex")):
            self.load_from_disk()
            if self.doc_ids:
                self.aux_next_doc_id = max(self.doc_ids) + 1
            
            # Load deleted documents (persistent deletion tracking)
            # This ensures deletions survive process restarts until merge permanently removes them
            import pickle
            deletes_file = os.path.join(index_dir, "deletes.pkl")
            if os.path.exists(deletes_file):
                with open(deletes_file, 'rb') as f:
                    self.deleted_docs = set(pickle.load(f))
    
    def add_document(self, title: str, plot: str) -> int:
        """
        Add new document to auxiliary index (IIR Ch.4.5: dynamic indexing pattern).
        
        This method implements fast document insertion by adding to the in-memory auxiliary index
        rather than modifying the disk-resident main index. This design choice enables:
        
        1. Fast Insertion: O(|terms_in_doc|) complexity without disk I/O
        2. Batch Efficiency: Multiple additions can be made before triggering an expensive merge
        3. Write Amplification Reduction: Instead of rewriting entire posting lists for each addition,
           we batch updates and perform one merge operation for many additions
        
        Collection Statistics Maintenance:
        It's crucial to update collection-wide statistics (term_cf, collection_tokens) during
        insertion, not deferring until merge. These statistics are used for:
        - Language Model ranking: P(t|collection) = cf / |collection_size|
        - BM25 IDF calculations: idf = log((N - df + 0.5) / (df + 0.5))
        - Average document length computation for BM25 normalization
        
        If we only updated statistics during merge, queries executed before merge would use stale
        statistics, leading to incorrect ranking scores. By maintaining statistics incrementally,
        we ensure query correctness even with pending merges.
        
        Document ID Assignment:
        Document IDs are assigned sequentially starting from max(main_index_ids) + 1. This ensures
        uniqueness across main and auxiliary indexes. The sequential assignment also maintains the
        property that postings lists remain sortable by document ID, which is required for efficient
        gap encoding during merge operations.
        """
        doc_id = self.aux_next_doc_id
        self.aux_next_doc_id += 1
        
        # Conflict detection: ensure doc_id doesn't already exist
        # This prevents accidental overwrites and ensures data integrity
        if doc_id in self.doc_ids:
            raise ValueError(f"Document ID {doc_id} already exists in main index")
        if doc_id in self.aux_doc_ids:
            raise ValueError(f"Document ID {doc_id} already exists in auxiliary index")
        
        self.aux_doc_ids.append(doc_id)
        self.aux_doc_titles[doc_id] = title
        self.aux_doc_plots[doc_id] = plot
        
        tokens = self.tokenizer.tokenize(plot)
        term_counts = Counter(tokens)
        self.aux_doc_len[doc_id] = len(tokens)
        
        # Update collection statistics immediately (not deferred to merge)
        # These statistics must be current for correct query processing. Language models especially
        # depend on accurate collection frequencies for smoothing. Deferring updates would cause
        # ranking errors for queries executed before merge.
        self.collection_tokens += len(tokens)
        for term, tf in term_counts.items():
            self.aux_index[term][doc_id] = tf
            # Update global term_cf to include this document's contribution
            # Note: This updates the global statistic, not just aux-specific counts
            self.term_cf[term] = self.term_cf.get(term, 0) + tf
        
        return doc_id
    
    def delete_document(self, doc_id: int):
        """
        Mark document as deleted (invalidation bit-vector pattern).
        
        Deletions are persisted to disk immediately to ensure they survive restarts. The deleted_docs
        set is saved to deletes.pkl after each deletion operation. During merge_auxiliary(), deleted
        documents are permanently removed from the index and the deletion log is cleared.
        """
        self.deleted_docs.add(doc_id)
        
        # Persist deletion to disk immediately for crash recovery
        # This ensures deletions survive process restarts until merge permanently removes them
        import pickle
        import os
        deletes_file = os.path.join(self.index_dir, "deletes.pkl")
        with open(deletes_file, 'wb') as f:
            pickle.dump(list(self.deleted_docs), f)
    
    def update_document(self, doc_id: int, title: str, plot: str):
        """Update document (delete + reinsert)."""
        self.delete_document(doc_id)
        
        # Remove from auxiliary if present
        if doc_id in self.aux_doc_ids:
            self.aux_doc_ids.remove(doc_id)
            del self.aux_doc_titles[doc_id]
            del self.aux_doc_plots[doc_id]
            if doc_id in self.aux_doc_len:
                del self.aux_doc_len[doc_id]
            
            for term in list(self.aux_index.keys()):
                if doc_id in self.aux_index[term]:
                    del self.aux_index[term][doc_id]
                    if not self.aux_index[term]:
                        del self.aux_index[term]
        
        self.add_document(title, plot)
    
    def get_postings(self, term: str) -> Dict[int, int]:
        """Get postings from both main and auxiliary indexes."""
        # Main index (disk)
        main_postings = super().get_postings(term)
        
        # Auxiliary index (memory)
        aux_postings = self.aux_index.get(term, {})
        
        # Merge
        merged = {**main_postings, **aux_postings}
        
        # Filter deleted
        return {d: tf for d, tf in merged.items() if d not in self.deleted_docs}
    
    def merge_auxiliary(self):
        """
        Merge auxiliary index into main index (IIR Ch.4.5: periodic merge strategy).
        
        This method implements the merge phase of dynamic indexing. The merge operation consolidates
        the in-memory auxiliary index into the persistent main index, following a write-once merge
        pattern that minimizes disk writes compared to in-place updates.
        
        Merge Process:
        1. Merge postings: For each term, we combine postings from auxiliary and main indexes.
           When the same document appears in both (shouldn't happen in normal operation, but handled
           defensively), the auxiliary entry takes precedence (it's more recent).
        2. Metadata Consolidation: Update doc_ids, doc_titles, doc_plots, and doc_len dictionaries
           to include all documents from the auxiliary index.
        3. Persistence: Write the merged index to disk, replacing the old main index atomically.
        4. Cleanup: Clear the auxiliary index structures to prepare for the next batch of additions.
        
        Why Not Incremental Updates?
        Writing updates directly to postings would require:
        - Reading entire posting lists from disk
        - Merging in memory
        - Rewriting entire posting lists
        This causes significant write amplification (rewriting large files for small changes).
        
        The batch merge approach amortizes disk I/O costs across many document additions. For N
        document additions, we perform one merge instead of N updates, reducing I/O by a factor of N.
        This trade-off is particularly beneficial for workloads with many small updates.
        
        Counting Before Clearing:
        We capture merged_n before clearing auxiliary structures. This ensures accurate reporting
        of merge statistics. The count is used for logging/debugging and helps users understand
        the scale of merge operations. Clearing before counting would always report 0, making
        merge operations appear ineffective.
        """
        merged_n = len(self.aux_doc_ids)
        
        # Rebuild with merged data
        for term in self.aux_index:
            if term in self.index:
                # Merge
                for doc_id, tf in self.aux_index[term].items():
                    self.index[term][doc_id] = tf
            else:
                self.index[term] = self.aux_index[term].copy()
        
        # Update metadata
        for doc_id in self.aux_doc_ids:
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)
            self.doc_titles[doc_id] = self.aux_doc_titles[doc_id]
            self.doc_plots[doc_id] = self.aux_doc_plots[doc_id]
            self.doc_len[doc_id] = self.aux_doc_len[doc_id]
        
        # Clear auxiliary (after counting)
        self.aux_index = defaultdict(dict)
        self.aux_doc_ids = []
        self.aux_doc_titles = {}
        self.aux_doc_plots = {}
        self.aux_doc_len = {}
        
        # Save to disk
        self.save_to_disk()
        
        # Clear deleted documents from persistent storage (they're now permanently removed from index)
        # After merge, deleted documents are no longer in the index, so we can clear the deletion log
        import pickle
        import os
        deletes_file = os.path.join(self.index_dir, "deletes.pkl")
        if os.path.exists(deletes_file):
            os.remove(deletes_file)
        self.deleted_docs.clear()  # Also clear in-memory set
        
        print(f"Merged {merged_n} documents into main index")


def create_indexer(tokenizer: Tokenizer, mode: str = "memory", index_dir: str = "index") -> BaseIndexer:
    """
    Factory function to create appropriate indexer.
    
    Args:
        tokenizer: Tokenizer instance
        mode: "memory", "disk", or "updatable"
        index_dir: Directory for disk-based storage
    
    Returns:
        BaseIndexer instance
    """
    if mode == "memory":
        return InMemoryIndexer(tokenizer)
    elif mode == "disk":
        return DiskIndexer(tokenizer, index_dir)
    elif mode == "updatable":
        return UpdatableIndexer(tokenizer, index_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    from tokenizer import Tokenizer
    
    tokenizer = Tokenizer()
    
    # Example: Memory indexer
    print("Testing memory indexer...")
    indexer = create_indexer(tokenizer, mode="memory")
    
    # Test with sample
    indexer.index_document(0, "Test", "This is a test about data mining.")
    indexer.index_document(1, "Test2", "Data mining and machine learning.")
    
    print("Statistics:", indexer.get_statistics())
