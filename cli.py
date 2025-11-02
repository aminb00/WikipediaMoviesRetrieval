#!/usr/bin/env python3
"""
CLI for Wikipedia Movies Retrieval System
Supports: memory, disk, and updatable indexing modes
"""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path

sys.path.append('Components')
from Tokenizer import tokenize
import Indexer
from QueryProcessor import QueryProcessor


def read_csv_documents(csv_path):
    """Read documents from CSV file(s). Returns list of (title, text) tuples."""
    documents = []
    
    # Check if it's a single file or directory
    if os.path.isfile(csv_path):
        csv_files = [csv_path]
    elif os.path.isdir(csv_path):
        csv_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) 
                     if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {csv_path}")
    else:
        # Try pattern matching (e.g., data/*.csv)
        csv_files = list(Path(csv_path).parent.glob(Path(csv_path).name)) if '*' in csv_path else []
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found at: {csv_path}")
    
    for csv_file in sorted(csv_files):
        print(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Handle different CSV schemas
        if 'title' in df.columns and 'plot' in df.columns:
            for _, row in df.iterrows():
                title = str(row['title'])
                plot = str(row.get('plot', ''))
                text = f"{title} {plot}".strip()
                if text:
                    documents.append((title, text))
        elif 'Title' in df.columns and 'Plot' in df.columns:
            for _, row in df.iterrows():
                title = str(row['Title'])
                plot = str(row.get('Plot', ''))
                text = f"{title} {plot}".strip()
                if text:
                    documents.append((title, text))
        else:
            # Try first two columns as title and plot
            cols = df.columns.tolist()
            if len(cols) >= 2:
                for _, row in df.iterrows():
                    title = str(row[cols[0]])
                    plot = str(row[cols[1]]) if len(cols) > 1 else ''
                    text = f"{title} {plot}".strip()
                    if text:
                        documents.append((title, text))
    
    print(f"Loaded {len(documents)} documents")
    return documents


def build_index_memory(csv_path, output_dir=None):
    """Build in-memory index (RAM + SPIMI)."""
    print("Building in-memory index (SPIMI)...")
    
    documents = read_csv_documents(csv_path)
    
    # Initialize memory indexer
    index_state = Indexer.init_memory(tokenize)
    
    # Index all documents
    for i, (title, text) in enumerate(documents, 1):
        Indexer.index_doc_mem(index_state, title, text)
        if i % 5000 == 0:
            print(f"  Indexed {i}/{len(documents)} documents...")
    
    print(f"✓ Indexed {len(documents)} documents")
    print(f"  Vocabulary size: {len(index_state['index']):,}")
    
    # Always save index state (default to current directory if not specified)
    if not output_dir:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    import pickle
    index_file = os.path.join(output_dir, 'index_memory.pkl')
    with open(index_file, 'wb') as f:
        pickle.dump(index_state, f)
    print(f"✓ Saved index to {index_file}")
    
    return index_state


def build_index_disk(csv_path, output_dir):
    """Build disk-based index (compression + lazy-load)."""
    print(f"Building disk-based index to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "terms"), exist_ok=True)
    
    # Initialize disk indexer
    index_state = Indexer.init_disk(output_dir, tokenize)
    
    # Create temporary folder with text files (Indexer.build_disk expects folder)
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        documents = read_csv_documents(csv_path)
        
        # Write documents as text files
        for i, (title, text) in enumerate(documents):
            file_path = os.path.join(temp_dir, f"doc_{i:06d}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Build disk index from temp folder
        Indexer.build_disk(index_state, temp_dir)
        
        print(f"✓ Built disk index with {len(documents)} documents")
        print(f"  Vocabulary size: {len(index_state['lex']):,}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return index_state


def build_index_updatable(csv_path, output_dir):
    """Build updatable index (main+aux)."""
    print(f"Building updatable index to {output_dir}...")
    
    # First build disk index
    disk_state = build_index_disk(csv_path, output_dir)
    
    # Then initialize updatable mode (loads existing disk index)
    upd_state = Indexer.init_upd(output_dir, tokenize, merge_threshold=100)
    
    print(f"✓ Built updatable index")
    return upd_state


def load_index(mode, index_dir):
    """Load existing index based on mode."""
    if mode == 'memory':
        import pickle
        with open(os.path.join(index_dir, 'index_memory.pkl'), 'rb') as f:
            return pickle.load(f)
    elif mode == 'disk':
        state = Indexer.init_disk(index_dir, tokenize)
        Indexer.load_disk_min(state)
        return state
    elif mode == 'updatable':
        return Indexer.init_upd(index_dir, tokenize)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_query_processor(index_state, mode):
    """Create QueryProcessor from index state, handling different modes."""
    # QueryProcessor expects index_state with 'index', 'doc_len', 'titles'
    # For disk/updatable, we need to adapt the state
    
    if mode == 'memory':
        # Direct use - QueryProcessor expects this format
        return QueryProcessor(index_state, k1=1.5, b=0.75)
    
    elif mode in ['disk', 'updatable']:
        # Need to adapt: QueryProcessor expects in-memory index dict
        # For disk/updatable, we need to convert to memory format
        # Load all postings into memory (for simplicity - could be optimized)
        
        print(f"Loading index into memory for query processing...")
        
        # Get postings function based on mode
        if mode == 'disk':
            get_postings_fn = Indexer.postings_disk
        else:  # updatable
            get_postings_fn = Indexer.postings_upd
        
        # Convert to memory format
        memory_index = {}
        terms = list(index_state.get('lex', {}).keys())
        total = len(terms)
        
        for i, term in enumerate(terms):
            if (i + 1) % 10000 == 0:
                print(f"  Loading postings: {i+1}/{total}...")
            postings = get_postings_fn(index_state, term)
            if postings:
                memory_index[term] = postings
        
        # Create adapted state in memory format
        adapted_state = {
            'index': memory_index,
            'doc_len': index_state.get('doc_len', {}),
            'titles': index_state.get('titles', {})
        }
        
        return QueryProcessor(adapted_state, k1=1.5, b=0.75)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description='Wikipedia Movies Retrieval System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build index')
    build_parser.add_argument('--mode', required=True, choices=['memory', 'disk', 'updatable'],
                             help='Index mode')
    build_parser.add_argument('--csv', required=True, 
                             help='CSV file or directory containing CSV files')
    build_parser.add_argument('--out', default=None,
                             help='Output directory (required for disk/updatable, optional for memory)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('--mode', required=True, choices=['memory', 'disk', 'updatable'],
                              help='Index mode')
    search_parser.add_argument('--model', required=True,
                              help='Ranking model: ltc.ltc, ntc.ltc, bm25, lm, etc.')
    search_parser.add_argument('--topk', type=int, default=10,
                              help='Number of results to return')
    search_parser.add_argument('--query', required=True,
                              help='Search query')
    search_parser.add_argument('--index', default=None,
                              help='Index directory (default: idx_<mode> for disk/updatable)')
    
    # Add command (updatable only)
    add_parser = subparsers.add_parser('add', help='Add document to updatable index')
    add_parser.add_argument('--mode', default='updatable', choices=['updatable'],
                           help='Mode (must be updatable)')
    add_parser.add_argument('--title', required=True,
                           help='Document title')
    add_parser.add_argument('--plot', required=True,
                           help='Document plot/text')
    add_parser.add_argument('--index', default='idx_upd',
                           help='Index directory')
    
    # Delete command (updatable only)
    delete_parser = subparsers.add_parser('delete', help='Delete document from updatable index')
    delete_parser.add_argument('--mode', default='updatable', choices=['updatable'],
                              help='Mode (must be updatable)')
    delete_parser.add_argument('--docid', type=int, required=True,
                              help='Document ID to delete')
    delete_parser.add_argument('--index', default='idx_upd',
                              help='Index directory')
    
    # Merge command (updatable only)
    merge_parser = subparsers.add_parser('merge', help='Merge auxiliary index into main')
    merge_parser.add_argument('--mode', default='updatable', choices=['updatable'],
                             help='Mode (must be updatable)')
    merge_parser.add_argument('--index', default='idx_upd',
                              help='Index directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'build':
            if args.mode == 'memory':
                build_index_memory(args.csv, args.out)
            elif args.mode == 'disk':
                if not args.out:
                    args.out = 'idx_disk'
                build_index_disk(args.csv, args.out)
            elif args.mode == 'updatable':
                if not args.out:
                    args.out = 'idx_upd'
                build_index_updatable(args.csv, args.out)
        
        elif args.command == 'search':
            # Determine index directory
            if args.index:
                index_dir = args.index
            elif args.mode == 'memory':
                index_dir = args.index or '.'
            elif args.mode == 'disk':
                index_dir = args.index or 'idx_disk'
            elif args.mode == 'updatable':
                index_dir = args.index or 'idx_upd'
            
            # Load index
            if args.mode == 'memory':
                index_file = os.path.join(index_dir, 'index_memory.pkl')
                if not os.path.exists(index_file):
                    print(f"Error: Memory index not found at {index_file}")
                    print("Please build the index first: python cli.py build --mode=memory --csv data/")
                    sys.exit(1)
                index_state = load_index(args.mode, index_dir)
            else:
                if not os.path.exists(index_dir):
                    print(f"Error: Index directory not found: {index_dir}")
                    print(f"Please build the index first: python cli.py build --mode={args.mode} --csv data/ --out {index_dir}")
                    sys.exit(1)
                index_state = load_index(args.mode, index_dir)
            
            # Get query processor
            qp = get_query_processor(index_state, args.mode)
            
            # Execute query
            if args.model == 'bm25':
                results = qp.compute_bm25_score(args.query)
            elif args.model == 'lm':
                results = qp.compute_lm_score(args.query, mu=2000, top_k=args.topk)
            else:
                # SMART notation (e.g., ltc.ltc, ntc.ltc)
                results = qp.rank_smart(args.query, weighting=args.model, top_k=args.topk)
            
            # Print results
            print(f"\nQuery: '{args.query}'")
            print(f"Model: {args.model}")
            print(f"Top {min(args.topk, len(results))} results:\n")
            for i, (title, score) in enumerate(results[:args.topk], 1):
                print(f"{i}. {title} (score: {score:.6f})")
        
        elif args.command == 'add':
            if args.mode != 'updatable':
                print("Error: 'add' command only works with updatable mode")
                sys.exit(1)
            
            index_state = load_index('updatable', args.index)
            doc_id = Indexer.add_upd(index_state, args.title, args.plot)
            print(f"✓ Added document (ID: {doc_id})")
        
        elif args.command == 'delete':
            if args.mode != 'updatable':
                print("Error: 'delete' command only works with updatable mode")
                sys.exit(1)
            
            index_state = load_index('updatable', args.index)
            Indexer.delete_upd(index_state, args.docid)
            print(f"✓ Deleted document (ID: {args.docid})")
        
        elif args.command == 'merge':
            if args.mode != 'updatable':
                print("Error: 'merge' command only works with updatable mode")
                sys.exit(1)
            
            index_state = load_index('updatable', args.index)
            Indexer.merge_upd(index_state)
            print("✓ Merged auxiliary index into main index")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

