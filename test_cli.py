#!/usr/bin/env python3
"""
Test script for CLI - checks if CLI commands work correctly
This script simulates CLI usage without requiring full dataset
"""

import sys
import os
import tempfile
import shutil

sys.path.append('Components')
from Tokenizer import tokenize
import Indexer

def test_cli_syntax():
    """Test if CLI can be imported and parsed"""
    try:
        # Try to compile CLI
        import py_compile
        py_compile.compile('cli.py', doraise=True)
        print("✓ CLI syntax is valid")
        return True
    except Exception as e:
        print(f"✗ CLI syntax error: {e}")
        return False

def test_memory_build():
    """Test memory index building"""
    try:
        index_state = Indexer.init_memory(tokenize)
        
        # Index a few test documents
        doc1 = Indexer.index_doc_mem(index_state, "Test Movie 1", "space adventure alien planet")
        doc2 = Indexer.index_doc_mem(index_state, "Test Movie 2", "murder mystery detective investigation")
        doc3 = Indexer.index_doc_mem(index_state, "Test Movie 3", "romantic love story comedy")
        
        assert doc1 == 0
        assert doc2 == 1
        assert doc3 == 2
        assert len(index_state['index']) > 0
        assert len(index_state['titles']) == 3
        
        print(f"✓ Memory build works: {len(index_state['index'])} terms, {len(index_state['titles'])} docs")
        return True, index_state
    except Exception as e:
        print(f"✗ Memory build error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_query_processing(index_state):
    """Test query processing"""
    try:
        from QueryProcessor import QueryProcessor
        
        qp = QueryProcessor(index_state, k1=1.5, b=0.75)
        
        # Test BM25
        bm25_results = qp.compute_bm25_score("space adventure")
        assert len(bm25_results) > 0
        print(f"✓ BM25 query works: {len(bm25_results)} results")
        
        # Test SMART
        smart_results = qp.rank_smart("space adventure", weighting="ltc.ltc", top_k=5)
        assert len(smart_results) > 0
        print(f"✓ SMART query works: {len(smart_results)} results")
        
        # Test Language Model
        lm_results = qp.compute_lm_score("space adventure", mu=2000, top_k=5)
        assert len(lm_results) > 0
        print(f"✓ Language Model query works: {len(lm_results)} results")
        
        return True
    except Exception as e:
        print(f"✗ Query processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_disk_build():
    """Test disk index building"""
    try:
        temp_dir = tempfile.mkdtemp()
        index_dir = os.path.join(temp_dir, "test_index")
        
        index_state = Indexer.init_disk(index_dir, tokenize)
        
        # Create temporary text files
        test_files_dir = os.path.join(temp_dir, "docs")
        os.makedirs(test_files_dir, exist_ok=True)
        
        with open(os.path.join(test_files_dir, "doc1.txt"), 'w') as f:
            f.write("space adventure alien planet")
        with open(os.path.join(test_files_dir, "doc2.txt"), 'w') as f:
            f.write("murder mystery detective")
        
        Indexer.build_disk(index_state, test_files_dir)
        
        assert len(index_state['lex']) > 0
        assert len(index_state['titles']) == 2
        
        print(f"✓ Disk build works: {len(index_state['lex'])} terms, {len(index_state['titles'])} docs")
        
        # Test loading
        Indexer.load_disk_min(index_state)
        assert 'lex' in index_state
        
        print(f"✓ Disk load works")
        
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"✗ Disk build error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_updatable():
    """Test updatable index"""
    try:
        temp_dir = tempfile.mkdtemp()
        index_dir = os.path.join(temp_dir, "test_upd_index")
        
        # First build disk index
        disk_state = Indexer.init_disk(index_dir, tokenize)
        test_files_dir = os.path.join(temp_dir, "docs")
        os.makedirs(test_files_dir, exist_ok=True)
        
        with open(os.path.join(test_files_dir, "doc1.txt"), 'w') as f:
            f.write("space adventure alien planet")
        
        Indexer.build_disk(disk_state, test_files_dir)
        
        # Then initialize updatable
        upd_state = Indexer.init_upd(index_dir, tokenize)
        
        # Test add
        doc_id = Indexer.add_upd(upd_state, "New Movie", "romantic love story")
        assert doc_id >= 0
        print(f"✓ Updatable add works: doc_id={doc_id}")
        
        # Test delete
        Indexer.delete_upd(upd_state, doc_id)
        print(f"✓ Updatable delete works")
        
        # Test merge
        Indexer.merge_upd(upd_state)
        print(f"✓ Updatable merge works")
        
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"✗ Updatable error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("CLI Test Suite")
    print("=" * 60)
    
    results = []
    
    print("\n[1/5] Testing CLI syntax...")
    results.append(("CLI Syntax", test_cli_syntax()))
    
    print("\n[2/5] Testing memory index build...")
    success, index_state = test_memory_build()
    results.append(("Memory Build", success))
    
    if index_state:
        print("\n[3/5] Testing query processing...")
        results.append(("Query Processing", test_query_processing(index_state)))
    
    print("\n[4/5] Testing disk index build...")
    results.append(("Disk Build", test_disk_build()))
    
    print("\n[5/5] Testing updatable index...")
    results.append(("Updatable", test_updatable()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

