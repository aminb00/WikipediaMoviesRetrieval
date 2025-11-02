# Wikipedia Movies Retrieval System

A compact Information Retrieval (IR) pipeline with SPIMI indexing, VSM/BM25/Language Model ranking.

## Quick Start

### Option 1: Docker (Recommended)

**Note:** Make sure Docker Desktop is running before proceeding.

```bash
# 1. Clone and build
git clone https://github.com/aminb00/WikipediaMoviesRetrieval.git
cd WikipediaMoviesRetrieval

# Check Docker is running
docker ps || echo "ERROR: Docker is not running. Start Docker Desktop first!"

# Build Docker image
docker build -t wikipedia-retrieval:latest .

# 2. Download dataset (one-time)
docker run --rm -v $(pwd)/data:/app/data wikipedia-retrieval:latest python download_dataset.py
```

### Option 2: Native Python (If Docker not available)

```bash
# Clone
git clone https://github.com/aminb00/WikipediaMoviesRetrieval.git
cd WikipediaMoviesRetrieval

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt

# Download dataset
python3 download_dataset.py
```

## Assignment Evaluation - Component Testing

Each component can be verified with a single command:

**Docker users:** Use the commands below.
**Native Python users:** Replace `docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python` with `python3` (or `python` if that works).

### 1. Tokenizer (3pts)
```bash
# Docker:
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python test_cli.py

# Native Python:
python3 test_cli.py
```
**Expected:** All tests pass, confirms tokenizer works.

### 2. Indexer RAM SPIMI (5pts)
```bash
# Docker:
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py build --mode=memory --csv data/
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "space adventure"

# Native Python:
python3 cli.py build --mode=memory --csv data/
python3 cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "space adventure"
```
**Expected:** Index builds successfully, query returns ranked results.

### 3. Indexer Disk + Lazy-load (+2pts)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py build --mode=disk --csv data/ --out idx_disk
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=disk --model=ltc.ltc --topk=5 --query "space adventure"
```
**Expected:** Disk index created in `idx_disk/`, query works.

### 4. Indexer Updatable (+2pts)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py build --mode=updatable --csv data/ --out idx_upd
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py add --mode=updatable --title "Test Film" --plot "A test movie about space exploration"
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=updatable --model=ltc.ltc --topk=5 --query "space exploration"
```
**Expected:** Index builds, document added, search finds new document.

### 5. Query Processor VSM ltc.ltc (4pts)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py build --mode=memory --csv data/
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using SMART ltc.ltc.

### 6. Query Processor SMART Variations (+2pts)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=ntc.ltc --topk=5 --query "romantic love story"
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=lnc.ltc --topk=5 --query "romantic love story"
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=atc.ltc --topk=5 --query "romantic love story"
```
**Expected:** All variations work (different scoring schemes).

### 7. Query Processor BM25 (+2pts)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=bm25 --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using BM25 ranking.

### 8. Query Processor Language Model (+2pts)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=lm --topk=5 --query "romantic love story"
```
**Expected:** Returns ranked results using Language Model (Dirichlet smoothing).

## All-in-One Test

```bash
# Docker:
docker run --rm -v $(pwd)/data:/app/data -v $(pwd):/app wikipedia-retrieval:latest python test_cli.py

# Native Python:
python3 test_cli.py
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
├── docker-compose.yml       # Docker Compose config
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
