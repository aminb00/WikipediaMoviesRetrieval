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
./run_docker.sh test_cli.py
```
**Expected:** All tests pass, confirms tokenizer works.

### 2. Indexer RAM SPIMI (5pts)
```bash
./run_docker.sh cli.py build --mode=memory --csv data/
./run_docker.sh cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "space adventure"
```
**Expected:** Index builds successfully, query returns ranked results.

### 3. Indexer Disk + Lazy-load (+2pts)
```bash
./run_docker.sh cli.py build --mode=disk --csv data/ --out idx_disk
./run_docker.sh cli.py search --mode=disk --model=ltc.ltc --topk=5 --query "space adventure"
```
**Expected:** Disk index created in `idx_disk/`, query works.

### 4. Indexer Updatable (+2pts)
```bash
./run_docker.sh cli.py build --mode=updatable --csv data/ --out idx_upd
./run_docker.sh cli.py add --mode=updatable --title "Test Film" --plot "A test movie about space exploration"
./run_docker.sh cli.py search --mode=updatable --model=ltc.ltc --topk=5 --query "space exploration"
```
**Expected:** Index builds, document added, search finds new document.

### 5. Query Processor VSM ltc.ltc (4pts)
```bash
./run_docker.sh cli.py build --mode=memory --csv data/
./run_docker.sh cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using SMART ltc.ltc.

### 6. Query Processor SMART Variations (+2pts)
```bash
./run_docker.sh cli.py search --mode=memory --model=ntc.ltc --topk=5 --query "romantic love story"
./run_docker.sh cli.py search --mode=memory --model=lnc.ltc --topk=5 --query "romantic love story"
./run_docker.sh cli.py search --mode=memory --model=atc.ltc --topk=5 --query "romantic love story"
```
**Expected:** All variations work (different scoring schemes).

### 7. Query Processor BM25 (+2pts)
```bash
./run_docker.sh cli.py search --mode=memory --model=bm25 --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using BM25 ranking.

### 8. Query Processor Language Model (+2pts)
```bash
./run_docker.sh cli.py search --mode=memory --model=lm --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using Language Model (Dirichlet smoothing).

## All-in-One Test

```bash
./run_docker.sh test_cli.py
```

**Expected output:**
```
✓ PASS: CLI Syntax
✓ PASS: Memory Build
✓ PASS: Query Processing
✓ PASS: Disk Build
✓ PASS: Updatable
✓ All tests passed!
```

## Project Structure

```
WikipediaMoviesRetrieval/
├── Dockerfile               # Docker containerization
├── run_docker.sh            # Docker runner script (use this for all commands)
├── cli.py                   # CLI interface (build, search, add, delete, merge)
├── test_cli.py              # Test suite
├── Components/
│   ├── Tokenizer.py         # Regex-based tokenizer
│   ├── Indexer.py           # SPIMI (memory, disk, updatable)
│   └── QueryProcessor.py    # VSM, BM25, Language Model
└── data/                    # Movie datasets (CSV files)
```

## Documentation

Full technical report: `Documentation/main.pdf`
