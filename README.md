# WikipediaMoviesRetrieval

A compact, inspectable Information Retrieval (IR) pipeline over the Wikipedia Movies dataset. We keep things simple and transparent; details and rationale are in the report.

## What's inside (high level)

- **Tokenizer**: Minimal regex-based normalization (11 lines, zero dependencies)
- **Indexer (SPIMI)**: Single-pass in-memory inverted index; plus disk variant and updatable mode
- **Query Processor**: Retrieves and ranks documents based on user queries
- **Main script**: Loads CSVs, EDA, tokenizes title+plot, builds index, prints stats and sample postings

For the full design, references, and diagrams, see the PDF in `Documentation/`.

## Dataset

- **Source**: Kaggle — exactful/wikipedia-movies
- **Schema**: `title,image,plot` (we index `title + plot`; `image` URL is ignored)
- **Size**: 17,830 movies across 6 decades
  - 1970s: 1,770 | 1980s: 2,338 | 1990s: 3,105
  - 2000s: 4,416 | 2010s: 4,960 | 2020s: 1,241
- **Files**: Decade CSVs in `Data/`

You can use the included `download_dataset.py` to fetch the CSVs (uses `kagglehub`). If already present, skip this step.

## System Performance

After indexing the complete dataset:
- **Documents Indexed**: 17,830 movie plots
- **Total Tokens**: 8,479,845 tokens processed
- **Vocabulary Size**: 92,857 unique terms
- **Total Postings**: 4,023,002 term-document pairs
- **Avg Tokens/Doc**: ~476 tokens
- **Index Density**: ~43 postings per term

These metrics demonstrate efficient indexing of a medium-sized corpus with rich vocabulary coverage.

## Requirements

- Python 3.10+
- Packages: `pandas` (and `kagglehub` if downloading dataset)

## Reproducibility

**Tested on:** Python 3.11+, macOS 14 / Ubuntu 22.04

**Status:** ✅ All CLI commands tested and working

### Setup

```bash
git clone https://github.com/aminb00/WikipediaMoviesRetrieval.git
cd WikipediaMoviesRetrieval
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download Dataset (if needed)

```bash
python download_dataset.py
```

This will download the Wikipedia Movies dataset from Kaggle to the `data/` folder.

### Test CLI (Optional)

Verify CLI is working correctly:

```bash
python test_cli.py
```

Expected output: All tests should pass (✓ PASS for all components).

### Build the index (RAM + SPIMI)

```bash
python cli.py build --mode=memory --csv data/
```

### Build the index (Disk + compression + lazy-load)

```bash
python cli.py build --mode=disk --csv data/ --out idx_disk
```

### Build the updatable index (main+aux)

```bash
python cli.py build --mode=updatable --csv data/ --out idx_upd
```

### Query examples

```bash
# SMART VSM queries
python cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "space adventure alien planet"
python cli.py search --mode=disk    --model=ntc.ltc --topk=5 --query "murder mystery detective"
python cli.py search --mode=disk    --model=lnc.ltc --topk=5 --query "romantic love story"

# BM25 ranking
python cli.py search --mode=memory --model=bm25 --topk=5 --query "romantic love story"

# Language Model (Dirichlet smoothing)
python cli.py search --mode=memory --model=lm --topk=5 --query "space adventure alien"
```

**Available ranking models:**
- **SMART notation**: `ltc.ltc`, `ntc.ltc`, `lnc.ltc`, `atc.ltc`, etc. 
  - Format: `query_scheme.document_scheme` (e.g., `ltc.ltc` = log tf, idf, cosine for both query and docs)
- **`bm25`**: BM25 ranking with k1=1.5, b=0.75
- **`lm`**: Language Model with Dirichlet smoothing (μ=2000)

**Note:** For `disk` and `updatable` modes, the index will be loaded into memory during query processing for optimal performance.

### Update workflow (updatable)

After building an updatable index, you can add, delete, and merge documents:

```bash
# Add a new document (automatically merges if threshold exceeded)
python cli.py add --mode=updatable --title "New Film" --plot "A detective on Mars investigating alien mysteries..."

# Delete a document by ID
python cli.py delete --mode=updatable --docid 1234

# Manually trigger merge (auxiliary index → main index)
python cli.py merge --mode=updatable

# Search after updates
python cli.py search --mode=updatable --model=ltc.ltc --topk=5 --query "detective mars"
```

**Important:** Documents added via `add` are stored in auxiliary (RAM) index. Use `merge` to persist them to disk. Auto-merge occurs when auxiliary index reaches 100 documents (configurable).

### Alternative: Run Full Pipeline (Legacy)

```bash
python main.py
```

The script will:
1. Load all decade CSVs from `data/`
2. Print dataset overview and EDA statistics
3. Tokenize `title + plot` using regex tokenizer
4. Build SPIMI inverted index (in-memory)
5. Display index statistics and sample postings for common terms
6. Run query processing demos with:
   - SMART ltc.ltc (VSM)
   - SMART ntc.ltc (VSM)
   - BM25 ranking

## Project Structure

```
WikipediaMoviesRetrieval/
├── cli.py                   # CLI interface (build, search, add, delete, merge)
├── main.py                  # Legacy entry point (full pipeline demo)
├── test_cli.py              # CLI test suite (verify all features work)
├── Components/
│   ├── Tokenizer.py         # Regex-based tokenizer (11 lines)
│   ├── Indexer.py           # SPIMI indexing (memory, disk, updatable)
│   └── QueryProcessor.py   # VSM, BM25, Language Model ranking
├── data/                    # Movie datasets by decade (CSV files)
├── Documentation/
│   ├── main.tex             # Technical report (LaTeX)
│   └── main.pdf             # Compiled report
├── download_dataset.py      # Dataset fetcher (kagglehub)
└── requirements.txt         # Python dependencies
```

## CLI Commands Reference

### Build Commands

| Command | Description |
|---------|-------------|
| `python cli.py build --mode=memory --csv data/` | Build in-memory index (fastest, RAM only) |
| `python cli.py build --mode=disk --csv data/ --out idx_disk` | Build disk-based index (compressed, lazy-load) |
| `python cli.py build --mode=updatable --csv data/ --out idx_upd` | Build updatable index (main + auxiliary) |

### Search Commands

| Command | Description |
|---------|-------------|
| `python cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "query"` | Search with SMART notation |
| `python cli.py search --mode=disk --model=bm25 --topk=5 --query "query"` | Search with BM25 |
| `python cli.py search --mode=memory --model=lm --topk=5 --query "query"` | Search with Language Model |

**Modes:**
- `memory`: In-memory index (fastest query, requires all data in RAM)
- `disk`: Disk-based index (compressed, loads terms on demand)
- `updatable`: Updatable index (supports add/delete/merge operations)

### Update Commands (Updatable Mode Only)

| Command | Description |
|---------|-------------|
| `python cli.py add --mode=updatable --title "Title" --plot "Plot text"` | Add new document |
| `python cli.py delete --mode=updatable --docid 1234` | Delete document by ID |
| `python cli.py merge --mode=updatable` | Merge auxiliary → main index |

## Testing

Run the test suite to verify all features:

```bash
python test_cli.py
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

## Documentation

This README is intentionally concise. Please see `Documentation/main.pdf` for the full report (motivation, design choices, and references).