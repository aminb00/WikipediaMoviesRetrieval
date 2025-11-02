# Wikipedia Movies Retrieval System

A compact Information Retrieval (IR) pipeline with SPIMI indexing, VSM/BM25/Language Model ranking.

## Quick Start (Docker)

```bash
# 1. Clone and build
git clone https://github.com/aminb00/WikipediaMoviesRetrieval.git
cd WikipediaMoviesRetrieval

# Build Docker image
docker build -t wikipedia-retrieval:latest .

# 2. Download dataset (one-time)
./run_docker.sh download_dataset.py
```

## Assignment Evaluation - Component Testing

Each component can be verified with a single command. Use `./run_docker.sh` prefix for all commands.

### 1. Tokenizer (3pts)
```bash
# Tokenize example text from dataset
./run_docker.sh cli.py tokenize --example

# Tokenize custom text
./run_docker.sh cli.py tokenize --text "A space adventure about aliens exploring distant planets in the galaxy."
```
**Expected:** 
- First command shows tokenized output from a dataset example
- Second command shows tokenized output of the provided text
- Output shows lowercase alphanumeric tokens only (e.g., "space", "adventure", "aliens", "exploring")

### 2. Indexer RAM SPIMI (5pts)
```bash
# Build memory index (used for sections 2, 7, 8, 9)
./run_docker.sh cli.py build --mode=memory --csv data/

# Test search with memory index
./run_docker.sh cli.py search --mode=memory --test=bm25 --topk=5 --query "space adventure"
```
**Expected:** Index builds successfully, query returns ranked results.

### 3. Indexer Disk + Lazy-load (+2pts)
```bash
# Build disk index
./run_docker.sh cli.py build --mode=disk --csv data/ --out idx_disk

# Test search with disk index
./run_docker.sh cli.py search --mode=disk --test=bm25 --topk=5 --query "space adventure"
```
**Expected:** Disk index created in `idx_disk/`, query works.

### 4. Indexer Updatable - Insert (+1pt)
```bash
# Build updatable index (used for sections 4, 5, 6)
./run_docker.sh cli.py build --mode=updatable --csv data/ --out idx_upd

# Add document
./run_docker.sh cli.py add --mode=updatable --title "Test Film" --plot "A test movie about space exploration"

# Search to verify
./run_docker.sh cli.py search --mode=updatable --test=bm25 --topk=5 --query "space exploration"
```
**Expected:** Index builds, document added, search finds new document.

### 5. Indexer Updatable - Update (+0.5pt)
```bash
# Update document (uses existing idx_upd from section 4)
./run_docker.sh cli.py update --mode=updatable --docid 100 --title "Updated Film" --plot "A completely new plot about time travel"

# Search to verify
./run_docker.sh cli.py search --mode=updatable --test=bm25 --topk=5 --query "time travel"
```
**Expected:** Document updated, search finds updated content.

### 6. Indexer Updatable - Delete (+0.5pt)
```bash
# Delete document (uses existing idx_upd from section 4)
./run_docker.sh cli.py delete --mode=updatable --docid 100

# Search to verify
./run_docker.sh cli.py search --mode=updatable --test=bm25 --topk=5 --query "romantic"
```
**Expected:** Document deleted, search confirms deletion (deleted doc doesn't appear).

### 7. Query Processor VSM ltc.ltc (4pts)
```bash
# Test LTC ranking (uses existing memory index from section 2)
./run_docker.sh cli.py search --mode=memory --test=ltc --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using SMART ltc.ltc.

### 8. Query Processor SMART Variations (+2pts)
```bash
# Test NTC ranking (uses existing memory index from section 2)
./run_docker.sh cli.py search --mode=memory --test=ntc --topk=5 --query "romantic love story"

# Test LNC ranking
./run_docker.sh cli.py search --mode=memory --test=lnc --topk=5 --query "romantic love story"

# Test ATC ranking
./run_docker.sh cli.py search --mode=memory --test=atc --topk=5 --query "romantic love story"
```
**Expected:** All variations work (different scoring schemes: ntc.ltc, lnc.ltc, atc.ltc).

### 9. Query Processor BM25 (+1pt)
```bash
# Test BM25 ranking (uses existing memory index from section 2)
./run_docker.sh cli.py search --mode=memory --test=bm25 --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using BM25 ranking.

### 10. Query Processor Language Model (+1pt)
```bash
# Test Language Model (uses existing memory index from section 2)
./run_docker.sh cli.py search --mode=memory --test=lm --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using Language Model (Dirichlet smoothing).

### 10. Memory vs Disk Mode Comparison (Optional - For Report)
```bash
# Compare memory (RAM) vs disk (lazy-load) indexing modes
# This demonstrates the difference between assignment parts (a) and (b)
./run_docker.sh cli.py compare --csv data/ --query "romantic love story" --test=ltc --topk=10
```
**Expected:** Side-by-side comparison table showing:
- Same query results from memory and disk modes
- Match rate at same ranks
- Storage size differences
- Key differences explanation

### 11. Quick Test - Compare All Models (Optional)
```bash
# Compare all models side-by-side (uses existing memory index from section 2)
./run_docker.sh cli.py test --mode=memory --query "romantic love story" --topk=5
```
**Expected:** Table showing LTC, BM25, and LM results side-by-side for easy comparison.

### 12. Complete Timeline Test - All Tasks (Recommended)
```bash
# Run all assignment tasks and show complete timeline
./run_docker.sh cli.py timeline --csv data/
```
**Expected:** 
- Tests all 9 assignment tasks in sequence
- Shows timeline table with Task, Operation, Query, Status, and Details
- Automatically builds required indexes and tests all components
- Final summary with pass/fail counts

## Available Test Types

- `--test=ltc`: SMART ltc.ltc (log tf, idf, cosine)
- `--test=ntc`: SMART ntc.ltc (natural tf, idf, cosine)
- `--test=lnc`: SMART lnc.ltc (log tf, no idf, cosine)
- `--test=atc`: SMART atc.ltc (augmented tf, idf, cosine)
- `--test=bm25`: BM25 ranking
- `--test=lm`: Language Model (Dirichlet smoothing)

## Project Structure

```
WikipediaMoviesRetrieval/
├── Dockerfile               # Docker containerization
├── run_docker.sh            # Docker runner script (use this for all commands)
├── cli.py                   # CLI interface (build, search, add, delete, update, merge)
├── test_cli.py              # Test suite
├── Components/
│   ├── Tokenizer.py         # Regex-based tokenizer
│   ├── Indexer.py           # SPIMI (memory, disk, updatable)
│   └── QueryProcessor.py    # VSM, BM25, Language Model
└── data/                    # Movie datasets (CSV files)
```

## Documentation

Full technical report: `Documentation/main.pdf`
