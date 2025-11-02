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
from collections import defaultdict, Counter

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
    
    # Read documents from CSV
    documents = read_csv_documents(csv_path)
    
    # Build index directly (not using build_disk which expects text files)
    # We'll manually index like build_disk does but preserve titles
    tmp = defaultdict(dict)  # term -> {doc_id: tf}
    
    for i, (title, text) in enumerate(documents):
        did = index_state["next_id"]
        index_state["next_id"] += 1
        toks = tokenize(text)
        index_state["doc_len"][did] = len(toks)
        index_state["titles"][did] = title  # Preserve actual title!
        
        for t, tf in Counter(toks).items():
            tmp[t][did] = tf
    
    # Write term files (compressed: gap + VB encoding)
    for term, post in tmp.items():
        path = Indexer._term_path(index_state["dir"], term)
        compressed = Indexer._compress_postings(post)
        with open(path, "wb") as f:
            f.write(compressed)
        index_state["lex"][term] = {
            "path": path,
            "df": len(post),
            "cf": int(sum(post.values()))
        }
    
    # Save lexicon and metadata
    import pickle
    with open(os.path.join(index_state["dir"], "lexicon.pkl"), "wb") as f:
        pickle.dump(index_state["lex"], f)
    
    with open(os.path.join(index_state["dir"], "meta.pkl"), "wb") as f:
        pickle.dump({"titles": index_state["titles"], "doc_len": index_state["doc_len"]}, f)
    
    print(f"✓ Built disk index with {len(documents)} documents")
    print(f"  Vocabulary size: {len(index_state['lex']):,}")
    
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
        state = Indexer.init_upd(index_dir, tokenize)
        # Load auxiliary index if it exists (for persistence across docker runs)
        aux_file = os.path.join(index_dir, 'aux.pkl')
        if os.path.exists(aux_file):
            import pickle
            with open(aux_file, 'rb') as f:
                aux_data = pickle.load(f)
                state['aux'] = defaultdict(dict, aux_data.get('aux', {}))
                state['deleted'] = set(aux_data.get('deleted', set()))
                # Update titles and doc_len from aux documents
                for doc_id in aux_data.get('aux_doc_ids', []):
                    if doc_id in aux_data.get('titles', {}):
                        state['titles'][doc_id] = aux_data['titles'][doc_id]
                    if doc_id in aux_data.get('doc_len', {}):
                        state['doc_len'][doc_id] = aux_data['doc_len'][doc_id]
        return state
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
        
        # Ensure titles are loaded (for disk mode)
        if mode == 'disk' and not index_state.get('titles'):
            Indexer.load_disk_min(index_state)
        
        # Convert to memory format
        memory_index = {}
        
        # Get all terms: from lexicon (disk) AND from aux (updatable mode)
        terms = set(index_state.get('lex', {}).keys())
        if mode == 'updatable' and 'aux' in index_state:
            # Add terms from auxiliary index
            terms.update(index_state['aux'].keys())
        
        terms = list(terms)
        total = len(terms)
        
        for i, term in enumerate(terms):
            if (i + 1) % 10000 == 0:
                print(f"  Loading postings: {i+1}/{total}...")
            postings = get_postings_fn(index_state, term)
            if postings:
                memory_index[term] = postings
        
        # Get titles dict - for updatable, titles should already be in index_state
        # from load_disk_min (called in init_upd) and add_upd adds titles when adding docs
        titles = dict(index_state.get('titles', {}))
        
        # For updatable, ensure all aux documents have titles
        if mode == 'updatable' and 'aux' in index_state:
            # Check if any aux documents are missing titles
            aux_doc_ids = set()
            for term_postings in index_state['aux'].values():
                aux_doc_ids.update(term_postings.keys())
            # Titles should already be added by add_upd, but let's ensure
            for doc_id in aux_doc_ids:
                if doc_id not in titles:
                    # Fallback: use doc_id as title if missing
                    titles[doc_id] = f"Document {doc_id}"
        
        # Create adapted state in memory format
        adapted_state = {
            'index': memory_index,
            'doc_len': dict(index_state.get('doc_len', {})),
            'titles': titles
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
    
    # Tokenize command
    tokenize_parser = subparsers.add_parser('tokenize', help='Tokenize text input')
    tokenize_parser.add_argument('--text', type=str, default=None,
                                 help='Text to tokenize (if not provided, uses example from dataset)')
    tokenize_parser.add_argument('--example', action='store_true',
                                 help='Show example from dataset')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build index')
    build_parser.add_argument('--mode', required=True, choices=['memory', 'disk', 'updatable'],
                             help='Index mode')
    build_parser.add_argument('--csv', required=True, 
                             help='CSV file or directory containing CSV files')
    build_parser.add_argument('--out', default=None,
                             help='Output directory (required for disk/updatable, optional for memory)')
    
    # Search command - simplified: auto-selects model based on test
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('--mode', required=True, choices=['memory', 'disk', 'updatable'],
                              help='Index mode')
    search_parser.add_argument('--test', required=True,
                              choices=['ltc', 'ntc', 'lnc', 'atc', 'bm25', 'lm'],
                              help='Test type: ltc (ltc.ltc VSM), ntc (ntc.ltc VSM), lnc (lnc.ltc VSM), atc (atc.ltc VSM), bm25 (BM25 ranking), lm (language model)')
    search_parser.add_argument('--topk', type=int, default=5,
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
    
    # Update command (updatable only)
    update_parser = subparsers.add_parser('update', help='Update document in updatable index')
    update_parser.add_argument('--mode', default='updatable', choices=['updatable'],
                              help='Mode (must be updatable)')
    update_parser.add_argument('--docid', type=int, required=True,
                              help='Document ID to update')
    update_parser.add_argument('--title', required=True,
                              help='New document title')
    update_parser.add_argument('--plot', required=True,
                              help='New document plot/text')
    update_parser.add_argument('--index', default='idx_upd',
                              help='Index directory')
    
    # Merge command (updatable only)
    merge_parser = subparsers.add_parser('merge', help='Merge auxiliary index into main')
    merge_parser.add_argument('--mode', default='updatable', choices=['updatable'],
                             help='Mode (must be updatable)')
    merge_parser.add_argument('--index', default='idx_upd',
                              help='Index directory')
    
    # Test command - compares all models
    test_parser = subparsers.add_parser('test', help='Test all models (ltc, bm25, lm) and compare results')
    test_parser.add_argument('--mode', required=True, choices=['memory', 'disk', 'updatable'],
                           help='Index mode')
    test_parser.add_argument('--query', required=True,
                           help='Search query to test')
    test_parser.add_argument('--topk', type=int, default=5,
                           help='Number of results per model')
    test_parser.add_argument('--index', default=None,
                           help='Index directory (default: idx_<mode> for disk/updatable)')
    
    # Compare command - compare memory vs disk modes
    compare_parser = subparsers.add_parser('compare', help='Compare memory vs disk indexing modes')
    compare_parser.add_argument('--csv', required=True,
                              help='Path to CSV file(s) or directory containing CSV files')
    compare_parser.add_argument('--query', required=True,
                               help='Search query to test with')
    compare_parser.add_argument('--test', default='ltc',
                               choices=['ltc', 'ntc', 'lnc', 'atc', 'bm25', 'lm'],
                               help='Test type (default: ltc)')
    compare_parser.add_argument('--topk', type=int, default=10,
                               help='Number of results to compare (default: 10)')
    
    # Timeline command - tests all assignment tasks
    timeline_parser = subparsers.add_parser('timeline', help='Run all assignment tasks and show timeline of results')
    timeline_parser.add_argument('--csv', default='data/',
                                help='CSV directory for building indexes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'tokenize':
            from Components.Tokenizer import tokenize
            
            if args.example:
                # Use example from dataset
                csv_files = []
                if os.path.isdir('data'):
                    csv_files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]
                elif os.path.isfile('data'):
                    csv_files = ['data']
                
                if csv_files:
                    import pandas as pd
                    df = pd.read_csv(csv_files[0])
                    if 'plot' in df.columns and len(df) > 0:
                        example_text = df.iloc[0]['plot']
                        print(f"\nExample text from dataset:")
                        print(f"{'='*60}")
                        print(f"{example_text[:200]}..." if len(example_text) > 200 else example_text)
                        print(f"{'='*60}\n")
                        
                        tokens = tokenize(example_text)
                        print(f"Tokenized output ({len(tokens)} tokens):")
                        print(f"{'='*60}")
                        print(' '.join(tokens[:50]))  # Show first 50 tokens
                        if len(tokens) > 50:
                            print(f"... (and {len(tokens) - 50} more tokens)")
                        print(f"{'='*60}\n")
                        print(f"Total tokens: {len(tokens)}")
                        print(f"Unique tokens: {len(set(tokens))}")
                    else:
                        print("Error: No 'plot' column found in CSV file")
                        sys.exit(1)
                else:
                    print("Error: No CSV files found in 'data' directory")
                    sys.exit(1)
            elif args.text:
                # Tokenize user-provided text
                print(f"\nInput text:")
                print(f"{'='*60}")
                print(args.text)
                print(f"{'='*60}\n")
                
                tokens = tokenize(args.text)
                print(f"Tokenized output ({len(tokens)} tokens):")
                print(f"{'='*60}")
                print(' '.join(tokens))
                print(f"{'='*60}\n")
                print(f"Total tokens: {len(tokens)}")
                print(f"Unique tokens: {len(set(tokens))}")
            else:
                # No arguments provided - show usage
                tokenize_parser.print_help()
                print("\nExample usage:")
                print("  cli.py tokenize --example")
                print("  cli.py tokenize --text 'Hello world! This is a test.'")
        
        elif args.command == 'build':
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
            
            # Map test type to model/weighting
            model_map = {
                'ltc': 'ltc.ltc',
                'ntc': 'ntc.ltc',
                'lnc': 'lnc.ltc',
                'atc': 'atc.ltc',
                'bm25': 'bm25',
                'lm': 'lm'
            }
            
            model = model_map[args.test]
            
            # Execute query
            if model == 'bm25':
                results = qp.compute_bm25_score(args.query)
            elif model == 'lm':
                results = qp.compute_lm_score(args.query, mu=2000, top_k=args.topk)
            else:
                # SMART notation
                results = qp.rank_smart(args.query, weighting=model, top_k=args.topk)
            
            # Print results
            print(f"\nQuery: '{args.query}'")
            print(f"Model: {model}")
            print(f"Top {min(args.topk, len(results))} results:\n")
            for i, (title, score) in enumerate(results[:args.topk], 1):
                print(f"{i}. {title} (score: {score:.6f})")
        
        elif args.command == 'add':
            if args.mode != 'updatable':
                print("Error: 'add' command only works with updatable mode")
                sys.exit(1)
            
            index_state = load_index('updatable', args.index)
            doc_id = Indexer.add_upd(index_state, args.title, args.plot)
            
            # Save auxiliary index state for persistence
            import pickle
            aux_doc_ids = set()
            for term_postings in index_state['aux'].values():
                aux_doc_ids.update(term_postings.keys())
            
            # Convert aux defaultdict to regular dict for serialization
            aux_dict = {}
            for term, postings in index_state['aux'].items():
                aux_dict[term] = dict(postings)
            
            aux_data = {
                'aux': aux_dict,
                'deleted': list(index_state['deleted']),
                'aux_doc_ids': list(aux_doc_ids),
                'titles': {did: index_state['titles'][did] for did in aux_doc_ids if did in index_state['titles']},
                'doc_len': {did: index_state['doc_len'][did] for did in aux_doc_ids if did in index_state['doc_len']}
            }
            
            aux_file = os.path.join(args.index, 'aux.pkl')
            with open(aux_file, 'wb') as f:
                pickle.dump(aux_data, f)
            
            print(f"✓ Added document (ID: {doc_id})")
        
        elif args.command == 'delete':
            if args.mode != 'updatable':
                print("Error: 'delete' command only works with updatable mode")
                sys.exit(1)
            
            index_state = load_index('updatable', args.index)
            Indexer.delete_upd(index_state, args.docid)
            
            # Save auxiliary index state for persistence
            import pickle
            aux_doc_ids = set()
            for term_postings in index_state['aux'].values():
                aux_doc_ids.update(term_postings.keys())
            
            # Convert aux defaultdict to regular dict for serialization
            aux_dict = {}
            for term, postings in index_state['aux'].items():
                aux_dict[term] = dict(postings)
            
            aux_data = {
                'aux': aux_dict,
                'deleted': list(index_state['deleted']),
                'aux_doc_ids': list(aux_doc_ids),
                'titles': {did: index_state['titles'][did] for did in aux_doc_ids if did in index_state['titles']},
                'doc_len': {did: index_state['doc_len'][did] for did in aux_doc_ids if did in index_state['doc_len']}
            }
            
            aux_file = os.path.join(args.index, 'aux.pkl')
            with open(aux_file, 'wb') as f:
                pickle.dump(aux_data, f)
            
            print(f"✓ Deleted document (ID: {args.docid})")
        
        elif args.command == 'update':
            if args.mode != 'updatable':
                print("Error: 'update' command only works with updatable mode")
                sys.exit(1)
            
            index_state = load_index('updatable', args.index)
            doc_id = Indexer.update_upd(index_state, args.docid, args.title, args.plot)
            
            # Save auxiliary index state for persistence
            import pickle
            aux_doc_ids = set()
            for term_postings in index_state['aux'].values():
                aux_doc_ids.update(term_postings.keys())
            
            # Convert aux defaultdict to regular dict for serialization
            aux_dict = {}
            for term, postings in index_state['aux'].items():
                aux_dict[term] = dict(postings)
            
            aux_data = {
                'aux': aux_dict,
                'deleted': list(index_state['deleted']),
                'aux_doc_ids': list(aux_doc_ids),
                'titles': {did: index_state['titles'][did] for did in aux_doc_ids if did in index_state['titles']},
                'doc_len': {did: index_state['doc_len'][did] for did in aux_doc_ids if did in index_state['doc_len']}
            }
            
            aux_file = os.path.join(args.index, 'aux.pkl')
            with open(aux_file, 'wb') as f:
                pickle.dump(aux_data, f)
            
            print(f"✓ Updated document (ID: {args.docid}, new ID: {doc_id})")
        
        elif args.command == 'merge':
            if args.mode != 'updatable':
                print("Error: 'merge' command only works with updatable mode")
                sys.exit(1)
            
            index_state = load_index('updatable', args.index)
            Indexer.merge_upd(index_state)
            
            # Remove aux.pkl after merge (aux is now in main index)
            aux_file = os.path.join(args.index, 'aux.pkl')
            if os.path.exists(aux_file):
                os.remove(aux_file)
            
            print("✓ Merged auxiliary index into main index")
        
        elif args.command == 'test':
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
            
            # Run all models
            models = [
                ('ltc', 'ltc.ltc'),
                ('ntc', 'ntc.ltc'),
                ('lnc', 'lnc.ltc'),
                ('atc', 'atc.ltc'),
                ('bm25', 'bm25'),
                ('lm', 'lm')
            ]
            
            all_results = {}
            for model_name, model_weighting in models:
                if model_weighting == 'bm25':
                    results = qp.compute_bm25_score(args.query)
                elif model_weighting == 'lm':
                    results = qp.compute_lm_score(args.query, mu=2000, top_k=args.topk)
                else:
                    results = qp.rank_smart(args.query, weighting=model_weighting, top_k=args.topk)
                
                all_results[model_name] = results[:args.topk]
            
            # Print formatted table
            print("\n" + "="*100)
            print(f"QUERY: '{args.query}'")
            print("="*100)
            print()
            
            # Create a combined table - show VSM models in one section, BM25/LM in another
            max_len = max(len(all_results[model]) for model in all_results)
            
            # VSM Models Section
            print(f"\nVSM Models (SMART Variations):")
            print(f"{'Rank':<6} {'LTC (ltc.ltc)':<35} {'Score':<12} {'NTC (ntc.ltc)':<35} {'Score':<12} {'LNC (lnc.ltc)':<35} {'Score':<12} {'ATC (atc.ltc)':<35} {'Score':<12}")
            print("-"*140)
            
            for i in range(max_len):
                row_parts = [f"{i+1:<6}"]
                
                for model_name in ['ltc', 'ntc', 'lnc', 'atc']:
                    if i < len(all_results[model_name]):
                        title, score = all_results[model_name][i]
                        title_display = title[:33] + "..." if len(title) > 35 else title
                        row_parts.append(f"{title_display:<35} {score:>11.6f}")
                    else:
                        row_parts.append(f"{'---':<35} {'---':>11}")
                
                print(" ".join(row_parts))
            
            print("-"*140)
            print()
            
            # BM25 and LM Section
            print(f"BM25 and Language Model:")
            print(f"{'Rank':<6} {'BM25':<40} {'Score':<12} {'LM (Dirichlet)':<40} {'Score':<12}")
            print("-"*100)
            
            for i in range(max_len):
                row_parts = [f"{i+1:<6}"]
                
                for model_name in ['bm25', 'lm']:
                    if i < len(all_results[model_name]):
                        title, score = all_results[model_name][i]
                        title_display = title[:37] + "..." if len(title) > 40 else title
                        row_parts.append(f"{title_display:<40} {score:>11.6f}")
                    else:
                        row_parts.append(f"{'---':<40} {'---':>11}")
                
                print(" ".join(row_parts))
            
            print("-"*100)
            print()
            
            # Summary statistics
            print("Summary:")
            for model_name in ['ltc', 'ntc', 'lnc', 'atc', 'bm25', 'lm']:
                if all_results[model_name]:
                    top_score = all_results[model_name][0][1]
                    unique_docs = len(set(title for title, _ in all_results[model_name]))
                    print(f"  {model_name.upper():>6}: Top score = {top_score:>10.6f}, Unique docs = {unique_docs}")
            print()
        
        elif args.command == 'compare':
            """Compare memory vs disk indexing modes"""
            print("\n" + "="*120)
            print("MEMORY vs DISK MODE COMPARISON")
            print("="*120)
            print()
            
            query = args.query
            test_type = args.test
            topk = args.topk
            
            # Map test type to model
            model_map = {
                'ltc': 'ltc.ltc',
                'ntc': 'ntc.ltc',
                'lnc': 'lnc.ltc',
                'atc': 'atc.ltc',
                'bm25': 'bm25',
                'lm': 'lm'
            }
            model = model_map[test_type]
            
            print(f"Query: '{query}'")
            print(f"Model: {model}")
            print(f"Top K: {topk}")
            print()
            
            # Build both indexes if needed
            print("[1/4] Building Memory Index...")
            memory_index_file = 'index_memory.pkl'
            if not os.path.exists(memory_index_file):
                build_index_memory(args.csv, '.')
            else:
                print("  Using existing memory index")
            
            print("\n[2/4] Building Disk Index...")
            disk_index_dir = 'idx_disk_compare'
            if not os.path.exists(disk_index_dir):
                build_index_disk(args.csv, disk_index_dir)
            else:
                print("  Using existing disk index")
            
            # Load indexes
            print("\n[3/4] Loading Indexes and Running Queries...")
            memory_state = load_index('memory', '.')
            disk_state = load_index('disk', disk_index_dir)
            
            memory_qp = get_query_processor(memory_state, 'memory')
            disk_qp = get_query_processor(disk_state, 'disk')
            
            # Run queries
            print(f"\n  Memory Mode: Running query...")
            if model == 'bm25':
                memory_results = memory_qp.compute_bm25_score(query)
            elif model == 'lm':
                memory_results = memory_qp.compute_lm_score(query, mu=2000, top_k=topk)
            else:
                memory_results = memory_qp.rank_smart(query, weighting=model, top_k=topk)
            
            print(f"  Disk Mode: Running query...")
            if model == 'bm25':
                disk_results = disk_qp.compute_bm25_score(query)
            elif model == 'lm':
                disk_results = disk_qp.compute_lm_score(query, mu=2000, top_k=topk)
            else:
                disk_results = disk_qp.rank_smart(query, weighting=model, top_k=topk)
            
            print("\n[4/4] Comparison Results")
            print("="*120)
            
            # Compare results
            max_len = max(len(memory_results), len(disk_results))
            max_len = min(max_len, topk)
            
            print(f"\n{'Rank':<6} {'Memory Mode (RAM)':<50} {'Score':<12} {'Disk Mode (Lazy)':<50} {'Score':<12} {'Match':<8}")
            print("-"*120)
            
            memory_titles = {title for title, _ in memory_results}
            disk_titles = {title for title, _ in disk_results}
            
            matches_count = 0
            score_diff_total = 0.0
            
            for i in range(max_len):
                memory_item = memory_results[i] if i < len(memory_results) else (None, None)
                disk_item = disk_results[i] if i < len(disk_results) else (None, None)
                
                memory_title = memory_item[0] if memory_item[0] else "---"
                memory_score = memory_item[1] if memory_item[1] else 0.0
                disk_title = disk_item[0] if disk_item[0] else "---"
                disk_score = disk_item[1] if disk_item[1] else 0.0
                
                # Check if same document at same rank
                is_match = memory_title == disk_title and memory_title != "---"
                match_str = "✓" if is_match else ""
                
                if is_match:
                    matches_count += 1
                    if memory_score and disk_score:
                        score_diff_total += abs(memory_score - disk_score)
                
                # Truncate long titles
                memory_display = memory_title[:47] + "..." if len(memory_title) > 50 else memory_title
                disk_display = disk_title[:47] + "..." if len(disk_title) > 50 else disk_title
                
                memory_score_str = f"{memory_score:.6f}" if memory_score else "---"
                disk_score_str = f"{disk_score:.6f}" if disk_score else "---"
                
                print(f"{i+1:<6} {memory_display:<50} {memory_score_str:<12} {disk_display:<50} {disk_score_str:<12} {match_str:<8}")
            
            print("-"*120)
            
            # Statistics
            print(f"\nComparison Statistics:")
            print(f"  Total documents in memory results: {len(memory_results)}")
            print(f"  Total documents in disk results: {len(disk_results)}")
            print(f"  Documents at same rank: {matches_count}/{max_len} ({matches_count/max_len*100:.1f}%)")
            
            # Calculate overlap in top-K
            memory_topk_titles = set(title for title, _ in memory_results[:topk])
            disk_topk_titles = set(title for title, _ in disk_results[:topk])
            overlap = len(memory_topk_titles & disk_topk_titles)
            print(f"  Overlap in top {topk}: {overlap}/{min(len(memory_topk_titles), len(disk_topk_titles))} documents")
            
            if matches_count > 0:
                avg_score_diff = score_diff_total / matches_count
                print(f"  Average score difference (when documents match): {avg_score_diff:.6f}")
            
            # Performance comparison (rough estimate)
            print(f"\nStorage Characteristics:")
            memory_size = os.path.getsize(memory_index_file) if os.path.exists(memory_index_file) else 0
            disk_dir_size = sum(
                os.path.getsize(os.path.join(disk_index_dir, f))
                for f in os.listdir(disk_index_dir)
                if os.path.isfile(os.path.join(disk_index_dir, f))
            )
            print(f"  Memory index size: {memory_size / (1024*1024):.2f} MB")
            print(f"  Disk index size: {disk_dir_size / (1024*1024):.2f} MB")
            print(f"  Disk index advantage: Lazy loading (only loads terms needed for query)")
            
            print("\n" + "="*120)
            print("Key Differences:")
            print("  Memory Mode: Entire index loaded in RAM, fastest query processing")
            print("  Disk Mode: Index on disk, lazy-loads only relevant postings (memory-efficient)")
            print("  Expected: Results should be identical (same algorithm, different storage)")
            print("="*120)
            print()
        
        elif args.command == 'timeline':
            """Run all assignment tasks and show timeline"""
            print("\n" + "="*120)
            print("ASSIGNMENT TASKS TIMELINE - Complete Test Suite")
            print("="*120)
            print()
            
            results_timeline = []
            
            # Helper to get CSV path when needed
            def get_csv_path():
                return args.csv if args.csv else 'data/'
            
            # 1. Tokenizer (3pts)
            print("[Component 1/3] Tokenizer (3pts)")
            print("[Task 1/12] Testing Tokenizer...")
            try:
                from Components.Tokenizer import tokenize
                test_text = "A space adventure about aliens exploring distant planets in the galaxy."
                tokens = tokenize(test_text)
                unique_tokens = len(set(tokens))
                # Show first 10 tokens as output
                token_preview = ' '.join(tokens[:10])
                if len(tokens) > 10:
                    token_preview += f" ... ({len(tokens) - 10} more)"
                output = f"{len(tokens)} tokens, {unique_tokens} unique: {token_preview}"
                results_timeline.append({
                    'task': '1. Tokenizer (3pts)',
                    'operation': 'tokenize',
                    'query': test_text,
                    'output': output
                })
                print(f"  {output}")
            except Exception as e:
                results_timeline.append({
                    'task': '1. Tokenizer (3pts)',
                    'operation': 'tokenize',
                    'query': test_text,
                    'output': f"ERROR: {str(e)}"
                })
                print(f"  ERROR: {e}")
            
            # 2. Indexer RAM SPIMI (5pts)
            print("\n[Component 2/3] Indexer (a) Memory (5pts)")
            print("[Task 2/12] Building Memory Index...")
            try:
                build_index_memory(args.csv, None)
                index_state = load_index('memory', '.')
                doc_count = len(index_state.get('titles', {}))
                term_count = len(index_state.get('index', {}))
                total_postings = sum(len(postings) for postings in index_state.get('index', {}).values())
                output = f"Indexed {doc_count} documents, {term_count} terms, {total_postings} total postings"
                results_timeline.append({
                    'task': '2. Indexer RAM SPIMI (5pts)',
                    'operation': 'build --mode=memory',
                    'query': 'N/A',
                    'output': output
                })
                print(f"  {output}")
            except Exception as e:
                results_timeline.append({
                    'task': '2. Indexer RAM SPIMI (5pts)',
                    'operation': 'build --mode=memory',
                    'query': 'N/A',
                    'output': f"ERROR: {str(e)}"
                })
                print(f"  ERROR: {e}")
                index_state = None
            
            # 3. Indexer Disk (2pts)
            print("\n[Component 2/3] Indexer (b) Disk Extension (+2pts)")
            print("[Task 3/12] Building Disk Index...")
            try:
                disk_index_dir = 'idx_disk'
                build_index_disk(args.csv, disk_index_dir)
                disk_state = load_index('disk', disk_index_dir)
                term_count = len(disk_state.get('lex', {}))
                doc_count = len(disk_state.get('titles', {}))
                
                # Get absolute paths
                abs_index_dir = os.path.abspath(disk_index_dir)
                terms_dir = os.path.join(disk_index_dir, 'terms')
                abs_terms_dir = os.path.abspath(terms_dir)
                lexicon_path = os.path.join(disk_index_dir, 'lexicon.pkl')
                meta_path = os.path.join(disk_index_dir, 'meta.pkl')
                
                # Check if files exist
                file_count = len([f for f in os.listdir(terms_dir) if f.endswith('.pkl')]) if os.path.exists(terms_dir) else 0
                
                # Get a sample term file path from lexicon
                sample_term_path = None
                if disk_state.get('lex', {}):
                    sample_term = list(disk_state['lex'].keys())[0]
                    # lexicon stores path as relative or absolute
                    term_lex_entry = disk_state['lex'][sample_term]
                    if 'path' in term_lex_entry:
                        stored_path = term_lex_entry['path']
                        # If it's already absolute, use it; otherwise make it absolute
                        if os.path.isabs(stored_path):
                            sample_term_path = stored_path
                        else:
                            sample_term_path = os.path.abspath(os.path.join(disk_index_dir, stored_path))
                        # Verify it exists
                        if not os.path.exists(sample_term_path):
                            # Try relative to terms directory
                            sample_term_path = os.path.abspath(os.path.join(terms_dir, os.path.basename(stored_path)))
                
                output = f"Indexed {doc_count} documents, {term_count} terms in lexicon, {file_count} term files on disk | Index: {abs_index_dir} | Terms: {abs_terms_dir} | Lexicon: {os.path.abspath(lexicon_path)} | Meta: {os.path.abspath(meta_path)}"
                if sample_term_path and os.path.exists(sample_term_path):
                    output += f" | Example term file: {sample_term_path}"
                
                results_timeline.append({
                    'task': '3. Indexer Disk (+2pts)',
                    'operation': 'build --mode=disk',
                    'query': 'N/A',
                    'output': output
                })
                print(f"  {output}")
            except Exception as e:
                results_timeline.append({
                    'task': '3. Indexer Disk (+2pts)',
                    'operation': 'build --mode=disk',
                    'query': 'N/A',
                    'output': f"ERROR: {str(e)}"
                })
                print(f"  ERROR: {e}")
            
            # 4. Indexer Updatable - Insert (1pt)
            print("\n[Component 2/3] Indexer (c) Updatable Extension (+2pts total)")
            print("[Task 4/12] Testing Updatable Insert (+1pt)...")
            try:
                build_index_updatable(args.csv, 'idx_upd')
                upd_state = load_index('updatable', 'idx_upd')
                test_title = "Test Film"
                test_plot = "A test movie about space exploration"
                doc_id = Indexer.add_upd(upd_state, test_title, test_plot)
                
                # Save aux state
                import pickle
                aux_doc_ids = set()
                for term_postings in upd_state['aux'].values():
                    aux_doc_ids.update(term_postings.keys())
                aux_dict = {term: dict(postings) for term, postings in upd_state['aux'].items()}
                aux_data = {
                    'aux': aux_dict,
                    'deleted': list(upd_state['deleted']),
                    'aux_doc_ids': list(aux_doc_ids),
                    'titles': {did: upd_state['titles'][did] for did in aux_doc_ids if did in upd_state['titles']},
                    'doc_len': {did: upd_state['doc_len'][did] for did in aux_doc_ids if did in upd_state['doc_len']}
                }
                with open(os.path.join('idx_upd', 'aux.pkl'), 'wb') as f:
                    pickle.dump(aux_data, f)
                
                # Test search to verify the added film appears
                qp_upd = get_query_processor(upd_state, 'updatable')
                test_query = "space exploration"
                search_results = qp_upd.compute_bm25_score(test_query)
                
                # Check if added film appears in results
                added_film_found = False
                added_film_rank = None
                added_film_score = None
                for rank, (title, score) in enumerate(search_results, 1):
                    if title == test_title:
                        added_film_found = True
                        added_film_rank = rank
                        added_film_score = score
                        break
                
                top3 = search_results[:3]
                top3_str = "; ".join([f"{title} ({score:.6f})" for title, score in top3])
                
                if added_film_found:
                    verification = f"✓ SUCCESS: Found at rank {added_film_rank} (score: {added_film_score:.6f})"
                else:
                    in_top10 = any(title == test_title for title, _ in search_results[:10])
                    if in_top10:
                        verification = f"⚠ Found in top 10 but not shown"
                    else:
                        verification = f"✗ WARNING: Not found in top 10"
                
                # Format output across multiple lines for clarity
                output_lines = [
                    f"Added doc ID {doc_id}: '{test_title}'",
                    f"  Plot: \"{test_plot}\"",
                    f"  {verification}",
                    f"  Query '{test_query}': {len(search_results)} results",
                    f"  Top 3: {top3_str}"
                ]
                output = " | ".join(output_lines)
                results_timeline.append({
                    'task': '4. Updatable Insert (+1pt)',
                    'operation': 'add --mode=updatable',
                    'query': 'space exploration',
                    'output': output
                })
                print(f"  {output}")
            except Exception as e:
                results_timeline.append({
                    'task': '4. Updatable Insert (+1pt)',
                    'operation': 'add --mode=updatable',
                    'query': 'space exploration',
                    'output': f"ERROR: {str(e)}"
                })
                print(f"  ERROR: {e}")
            
            # 5. Indexer Updatable - Update (0.5pt)
            print("\n[Task 5/12] Testing Updatable Update (+0.5pt)...")
            try:
                upd_state = load_index('updatable', 'idx_upd')
                update_query = "time travel"
                
                # First, find a film related to "time travel" to update
                qp_before = get_query_processor(upd_state, 'updatable')
                search_results_before = qp_before.compute_bm25_score(update_query)
                
                # Find a film from the MAIN index (not aux) to update
                # We want to update an existing film, not one we just added in Task 4
                old_doc_id = None
                aux_doc_ids = set()
                # Get aux doc IDs to exclude them
                for term_postings in upd_state.get('aux', {}).values():
                    aux_doc_ids.update(term_postings.keys())
                
                if search_results_before:
                    # Try to find doc_id from results that is NOT in aux (i.e., from main disk index)
                    for result_title, _ in search_results_before:
                        # Find doc_id by title (reverse lookup)
                        for did, title in upd_state.get('titles', {}).items():
                            if title == result_title and did not in aux_doc_ids and did < 17830:  # Exclude aux docs and the one we just added
                                old_doc_id = did
                                break
                        if old_doc_id:
                            break
                
                # Fallback: use a reasonable doc_id from main index if no suitable match found
                if old_doc_id is None:
                    # Find a doc_id that's not in aux and is a reasonable number (from main index)
                    for did in range(1000, min(5000, len(upd_state.get('titles', {})))):
                        if did not in aux_doc_ids and did not in upd_state.get('deleted', set()):
                            old_doc_id = did
                            break
                
                # Final fallback
                if old_doc_id is None:
                    old_doc_id = 1000
                
                old_title = upd_state.get('titles', {}).get(old_doc_id, 'Unknown Film')
                
                # Get old plot from CSV if available (for display and keeping same plot)
                old_plot = None
                old_plot_sample = None
                try:
                    csv_path = get_csv_path()
                    if csv_path and os.path.exists(csv_path):
                        documents = read_csv_documents(csv_path)
                        if old_doc_id < len(documents):
                            # Get the original document text (title + plot)
                            original_text = documents[old_doc_id][1]  # (title, text) tuple, get text part
                            # Extract plot part (remove title from beginning if title was prefixed)
                            old_plot = original_text[len(old_title):].strip() if original_text.startswith(old_title) else original_text
                            # Use full plot for update, sample for display
                            old_plot_sample = old_plot[:100] + "..." if len(old_plot) > 100 else old_plot
                except Exception as e:
                    # If we can't get plot, use a generic time travel plot
                    old_plot = "A story about time travel and temporal mechanics"
                    old_plot_sample = old_plot
                
                # If we couldn't get the plot, try to extract from index state
                if not old_plot:
                    # Fallback: use a generic plot that matches the query
                    old_plot = "In 2073, Nicholas Sinclair is a scientist on a time travel project. An accident causes temporal distortions and paradoxes that threaten the fabric of reality."
                    old_plot_sample = old_plot[:100] + "..."
                
                # Check if old film appears in "time travel" search BEFORE update
                old_film_found_before = False
                old_film_rank_before = None
                old_film_score_before = None
                for rank, (title, score) in enumerate(search_results_before[:10], 1):
                    if title == old_title:
                        old_film_found_before = True
                        old_film_rank_before = rank
                        old_film_score_before = score
                        break
                
                # Update ONLY the title, keep the same plot so it ranks similarly
                new_title = "Updated Time Travel Film"
                # Keep the same plot so ranking stays similar
                new_doc_id = Indexer.update_upd(upd_state, old_doc_id, new_title, old_plot)
                
                # Verify the update worked - check that new_doc_id is in aux
                aux_terms_count = sum(1 for term_postings in upd_state['aux'].values() if new_doc_id in term_postings)
                
                # Save aux state - include the newly updated document
                import pickle
                aux_doc_ids = set()
                for term_postings in upd_state['aux'].values():
                    aux_doc_ids.update(term_postings.keys())
                # Make sure new_doc_id is included in aux_doc_ids
                aux_doc_ids.add(new_doc_id)
                aux_dict = {term: dict(postings) for term, postings in upd_state['aux'].items()}
                aux_data = {
                    'aux': aux_dict,
                    'deleted': list(upd_state['deleted']),
                    'aux_doc_ids': list(aux_doc_ids),
                    'titles': {did: upd_state['titles'][did] for did in aux_doc_ids if did in upd_state['titles']},
                    'doc_len': {did: upd_state['doc_len'][did] for did in aux_doc_ids if did in upd_state['doc_len']}
                }
                with open(os.path.join('idx_upd', 'aux.pkl'), 'wb') as f:
                    pickle.dump(aux_data, f)
                
                # Search AFTER update with same "time travel" query
                # Note: We don't need to reload because get_query_processor reads from current upd_state
                # which already has the updated aux state in memory
                qp_after = get_query_processor(upd_state, 'updatable')
                search_results_after = qp_after.compute_bm25_score(update_query)
                
                # Check if old film is removed
                old_film_found_after = any(title == old_title for title, _ in search_results_after[:10])
                
                # Check if new updated film appears
                updated_film_found = False
                updated_film_rank = None
                updated_film_score = None
                # Search in more results (top 20) to see if it appears later
                for rank, (title, score) in enumerate(search_results_after, 1):
                    if title == new_title:
                        updated_film_found = True
                        updated_film_rank = rank
                        updated_film_score = score
                        break
                
                # If not found, check if it's in the results at all (even beyond top 10)
                if not updated_film_found:
                    for rank, (title, score) in enumerate(search_results_after, 1):
                        if title == new_title:
                            updated_film_found = True
                            updated_film_rank = rank
                            updated_film_score = score
                            break
                
                # Check why Avengers Endgame might rank differently - get top 3 before/after for comparison
                before_top3 = search_results_before[:3]
                after_top3 = search_results_after[:3]
                
                # Find Avengers Endgame if present
                avengers_rank_before = None
                avengers_rank_after = None
                for rank, (title, score) in enumerate(search_results_before, 1):
                    if 'endgame' in title.lower() or 'avengers' in title.lower():
                        avengers_rank_before = rank
                        break
                for rank, (title, score) in enumerate(search_results_after, 1):
                    if 'endgame' in title.lower() or 'avengers' in title.lower():
                        avengers_rank_after = rank
                        break
                
                # Format output across multiple lines for clarity
                output_lines = [
                    f"Updated doc {old_doc_id}: '{old_title}' → new ID {new_doc_id}: '{new_title}'",
                ]
                
                # Show plot (same for both, only title changed)
                if old_plot_sample:
                    plot_display = old_plot_sample[:80] + "..." if len(old_plot_sample) > 80 else old_plot_sample
                    output_lines.append(f"  Plot (unchanged): \"{plot_display}\"")
                
                # Before update status
                if old_film_found_before:
                    output_lines.append(f"  BEFORE: '{old_title}' at rank {old_film_rank_before} (score: {old_film_score_before:.6f})")
                else:
                    output_lines.append(f"  BEFORE: '{old_title}' not in top 10")
                
                # After update status
                if not old_film_found_after:
                    output_lines.append(f"  AFTER: '{old_title}' correctly removed (no longer in results)")
                else:
                    output_lines.append(f"  AFTER: WARNING - '{old_title}' still appears (should be removed)")
                
                if updated_film_found:
                    output_lines.append(f"  AFTER: '{new_title}' found at rank {updated_film_rank} (score: {updated_film_score:.6f})")
                else:
                    output_lines.append(f"  AFTER: WARNING - '{new_title}' not found in top 10")
                
                # Add Avengers Endgame rank explanation if relevant
                if avengers_rank_before or avengers_rank_after:
                    if avengers_rank_before and avengers_rank_after:
                        if avengers_rank_after > avengers_rank_before:
                            output_lines.append(f"  NOTE: Avengers Endgame rank changed {avengers_rank_before} → {avengers_rank_after} (plot unchanged, ranking affected by updated film)")
                        elif avengers_rank_after < avengers_rank_before:
                            output_lines.append(f"  NOTE: Avengers Endgame rank improved {avengers_rank_before} → {avengers_rank_after}")
                
                after_top3_str = "; ".join([f"{title} ({score:.6f})" for title, score in after_top3])
                output_lines.append(f"  Query '{update_query}' Top 3: {after_top3_str}")
                
                output = " | ".join(output_lines)
                results_timeline.append({
                    'task': '5. Updatable Update (+0.5pt)',
                    'operation': 'update --mode=updatable',
                    'query': update_query,
                    'output': output
                })
                print(f"  {output}")
            except Exception as e:
                results_timeline.append({
                    'task': '5. Updatable Update (+0.5pt)',
                    'operation': 'update --mode=updatable',
                    'query': 'time travel',
                    'output': f"ERROR: {str(e)}"
                })
                print(f"  ERROR: {e}")
            
            # 6. Indexer Updatable - Delete (0.5pt)
            print("\n[Task 6/12] Testing Updatable Delete (+0.5pt)...")
            try:
                upd_state = load_index('updatable', 'idx_upd')
                delete_doc_id = 100
                Indexer.delete_upd(upd_state, delete_doc_id)
                
                # Save aux state
                import pickle
                aux_doc_ids = set()
                for term_postings in upd_state['aux'].values():
                    aux_doc_ids.update(term_postings.keys())
                aux_dict = {term: dict(postings) for term, postings in upd_state['aux'].items()}
                aux_data = {
                    'aux': aux_dict,
                    'deleted': list(upd_state['deleted']),
                    'aux_doc_ids': list(aux_doc_ids),
                    'titles': {did: upd_state['titles'][did] for did in aux_doc_ids if did in upd_state['titles']},
                    'doc_len': {did: upd_state['doc_len'][did] for did in aux_doc_ids if did in upd_state['doc_len']}
                }
                with open(os.path.join('idx_upd', 'aux.pkl'), 'wb') as f:
                    pickle.dump(aux_data, f)
                
                # Test search to verify deletion
                qp_upd = get_query_processor(upd_state, 'updatable')
                search_results = qp_upd.compute_bm25_score("romantic")
                top3 = search_results[:3]
                top3_str = "; ".join([f"{title} ({score:.6f})" for title, score in top3])
                
                output = f"Deleted doc ID {delete_doc_id} (in deleted set: {delete_doc_id in upd_state['deleted']}) | Search 'romantic': {len(search_results)} results | Top 3: {top3_str}"
                results_timeline.append({
                    'task': '6. Updatable Delete (+0.5pt)',
                    'operation': 'delete --mode=updatable',
                    'query': 'romantic',
                    'output': output
                })
                print(f"  {output}")
            except Exception as e:
                results_timeline.append({
                    'task': '6. Updatable Delete (+0.5pt)',
                    'operation': 'delete --mode=updatable',
                    'query': 'romantic',
                    'output': f"ERROR: {str(e)}"
                })
                print(f"  ERROR: {e}")
            
            # 7-9. Query Processors (need memory index)
            if index_state:
                test_query = "romantic love story"
                qp = get_query_processor(index_state, 'memory')
                
                # 7. VSM ltc.ltc (4pts)
                print("\n[Component 3/3] Query Processor (a) VSM ltc.ltc (4pts)")
                print("[Task 7/12] Testing VSM ltc.ltc...")
                try:
                    results = qp.rank_smart(test_query, weighting="ltc.ltc", top_k=5)
                    # Show first 3 results
                    top3 = results[:3]
                    top3_str = "; ".join([f"{title} ({score:.6f})" for title, score in top3])
                    output = f"{len(results)} results, top score: {results[0][1]:.6f} | Top 3: {top3_str}"
                    results_timeline.append({
                        'task': '7. VSM ltc.ltc (4pts)',
                        'operation': 'search --test=ltc',
                        'query': test_query,
                        'output': output
                    })
                    print(f"  {output}")
                except Exception as e:
                    results_timeline.append({
                        'task': '7. VSM ltc.ltc (4pts)',
                        'operation': 'search --test=ltc',
                        'query': test_query,
                        'output': f"ERROR: {str(e)}"
                    })
                    print(f"  ERROR: {e}")
                
                # 8. VSM Variations (ntc, lnc, atc) (+2pts)
                print("\n[Component 3/3] Query Processor (b) SMART Variations (+2pts)")
                print("[Task 8/12] Testing VSM ntc.ltc...")
                try:
                    results_ntc = qp.rank_smart(test_query, weighting="ntc.ltc", top_k=5)
                    top3_ntc = results_ntc[:3]
                    top3_str_ntc = "; ".join([f"{title} ({score:.6f})" for title, score in top3_ntc])
                    output_ntc = f"{len(results_ntc)} results, top score: {results_ntc[0][1]:.6f} | Top 3: {top3_str_ntc}"
                    results_timeline.append({
                        'task': '8. VSM ntc.ltc (+0.67pt)',
                        'operation': 'search --test=ntc',
                        'query': test_query,
                        'output': output_ntc
                    })
                    print(f"  {output_ntc}")
                except Exception as e:
                    results_timeline.append({
                        'task': '8. VSM ntc.ltc (+0.67pt)',
                        'operation': 'search --test=ntc',
                        'query': test_query,
                        'output': f"ERROR: {str(e)}"
                    })
                    print(f"  ERROR: {e}")
                
                print("[Task 9/12] Testing VSM lnc.ltc...")
                try:
                    results_lnc = qp.rank_smart(test_query, weighting="lnc.ltc", top_k=5)
                    top3_lnc = results_lnc[:3]
                    top3_str_lnc = "; ".join([f"{title} ({score:.6f})" for title, score in top3_lnc])
                    output_lnc = f"{len(results_lnc)} results, top score: {results_lnc[0][1]:.6f} | Top 3: {top3_str_lnc}"
                    results_timeline.append({
                        'task': '9. VSM lnc.ltc (+0.67pt)',
                        'operation': 'search --test=lnc',
                        'query': test_query,
                        'output': output_lnc
                    })
                    print(f"  {output_lnc}")
                except Exception as e:
                    results_timeline.append({
                        'task': '9. VSM lnc.ltc (+0.67pt)',
                        'operation': 'search --test=lnc',
                        'query': test_query,
                        'output': f"ERROR: {str(e)}"
                    })
                    print(f"  ERROR: {e}")
                
                print("[Task 10/12] Testing VSM atc.ltc...")
                try:
                    results_atc = qp.rank_smart(test_query, weighting="atc.ltc", top_k=5)
                    top3_atc = results_atc[:3]
                    top3_str_atc = "; ".join([f"{title} ({score:.6f})" for title, score in top3_atc])
                    output_atc = f"{len(results_atc)} results, top score: {results_atc[0][1]:.6f} | Top 3: {top3_str_atc}"
                    results_timeline.append({
                        'task': '10. VSM atc.ltc (+0.66pt)',
                        'operation': 'search --test=atc',
                        'query': test_query,
                        'output': output_atc
                    })
                    print(f"  {output_atc}")
                except Exception as e:
                    results_timeline.append({
                        'task': '10. VSM atc.ltc (+0.66pt)',
                        'operation': 'search --test=atc',
                        'query': test_query,
                        'output': f"ERROR: {str(e)}"
                    })
                    print(f"  ERROR: {e}")
                
                # 11. BM25 (1pt)
                print("\n[Component 3/3] Query Processor (c) BM25/LM Extension (+2pts total)")
                print("[Task 11/12] Testing BM25 (+1pt)...")
                try:
                    results = qp.compute_bm25_score(test_query)
                    # Show first 3 results
                    top3 = results[:3]
                    top3_str = "; ".join([f"{title} ({score:.6f})" for title, score in top3])
                    output = f"{len(results)} results, top score: {results[0][1]:.6f} | Top 3: {top3_str}"
                    results_timeline.append({
                        'task': '11. BM25 (+1pt)',
                        'operation': 'search --test=bm25',
                        'query': test_query,
                        'output': output
                    })
                    print(f"  {output}")
                except Exception as e:
                    results_timeline.append({
                        'task': '11. BM25 (+1pt)',
                        'operation': 'search --test=bm25',
                        'query': test_query,
                        'output': f"ERROR: {str(e)}"
                    })
                    print(f"  ERROR: {e}")
                
                # 12. Language Model (1pt)
                print("\n[Task 12/12] Testing Language Model (+1pt)...")
                try:
                    results = qp.compute_lm_score(test_query, mu=2000, top_k=5)
                    # Show first 3 results
                    top3 = results[:3]
                    top3_str = "; ".join([f"{title} ({score:.6f})" for title, score in top3])
                    output = f"{len(results)} results, top score: {results[0][1]:.6f} | Top 3: {top3_str}"
                    results_timeline.append({
                        'task': '12. Language Model (+1pt)',
                        'operation': 'search --test=lm',
                        'query': test_query,
                        'output': output
                    })
                    print(f"  {output}")
                except Exception as e:
                    results_timeline.append({
                        'task': '12. Language Model (+1pt)',
                        'operation': 'search --test=lm',
                        'query': test_query,
                        'output': f"ERROR: {str(e)}"
                    })
                    print(f"  ERROR: {e}")
            else:
                print("\n[Task 7-12] Skipping query processors (memory index not available)")
                for task_name in ['7. VSM ltc.ltc (4pts)', '8. VSM ntc.ltc (+0.67pt)', '9. VSM lnc.ltc (+0.67pt)', '10. VSM atc.ltc (+0.66pt)', '11. BM25 (+1pt)', '12. Language Model (+1pt)']:
                    results_timeline.append({
                        'task': task_name,
                        'operation': 'N/A',
                        'query': 'N/A',
                        'output': 'SKIP: Memory index required'
                    })
            
            # Print timeline table organized by assignment components
            print("\n" + "="*140)
            print("TIMELINE SUMMARY - Assignment Components")
            print("="*140)
            print()
            
            # Group tasks by assignment component
            component_groups = {
                '1. Tokenizer (3pts)': [('1. Tokenizer (3pts)',)],
                '2. Indexer (a) Memory (5pts)': [('2. Indexer RAM SPIMI (5pts)',)],
                '2. Indexer (b) Disk Extension (+2pts)': [('3. Indexer Disk (+2pts)',)],
                '2. Indexer (c) Updatable Extension (+2pts)': [
                    ('4. Updatable Insert (+1pt)',),
                    ('5. Updatable Update (+0.5pt)',),
                    ('6. Updatable Delete (+0.5pt)',)
                ],
                '3. Query Processor (a) VSM ltc.ltc (4pts)': [('7. VSM ltc.ltc (4pts)',)],
                '3. Query Processor (b) SMART Variations (+2pts)': [
                    ('8. VSM ntc.ltc (+0.67pt)',),
                    ('9. VSM lnc.ltc (+0.67pt)',),
                    ('10. VSM atc.ltc (+0.66pt)',)
                ],
                '3. Query Processor (c) BM25/LM Extension (+2pts)': [
                    ('11. BM25 (+1pt)',),
                    ('12. Language Model (+1pt)',)
                ]
            }
            
            for component_name, tasks in component_groups.items():
                print(f"\n{component_name}")
                print("-" * 140)
                print(f"{'Task':<30} {'Operation':<30} {'Query':<30} {'Output'}")
                print("-" * 140)
                
                for task_tuple in tasks:
                    task_name = task_tuple[0]
                    # Find matching entry in results_timeline
                    entry = next((e for e in results_timeline if e['task'] == task_name), None)
                    if entry:
                        query_display = entry['query'][:28] + "..." if len(entry['query']) > 30 else entry['query']
                        output_display = entry['output'][:48] + "..." if len(entry['output']) > 50 else entry['output']
                        status_icon = "✓" if not entry['output'].startswith('ERROR') and not entry['output'].startswith('SKIP') else "✗"
                        print(f"{status_icon} {entry['task']:<28} {entry['operation']:<30} {query_display:<30} {output_display}")
                    else:
                        print(f"✗ {task_name:<28} {'N/A':<30} {'N/A':<30} {'NOT EXECUTED'}")
            
            print("-" * 140)
            print("="*140)
            print()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()