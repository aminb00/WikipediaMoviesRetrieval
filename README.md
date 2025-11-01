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


## Run

```bash
python main.py
```

The script will:
1. Load all decade CSVs from `Data/`
2. Print dataset overview and EDA statistics
3. Tokenize `title + plot` using regex tokenizer
4. Build SPIMI inverted index (in-memory)
5. Display index statistics and sample postings for common terms

## Project Structure

```
WikipediaMoviesRetrieval/
├── main.py                  # Entry point
├── Components/
│   ├── Tokenizer.py         # Regex-based tokenizer (11 lines)
│   └── Indexer.py           # SPIMI indexing (memory, disk, updatable)
├── Data/                    # Movie datasets by decade (CSV files)
├── Documentation/
│   ├── main.tex             # Technical report (LaTeX)
│   └── main.pdf             # Compiled report
└── download_dataset.py      # Dataset fetcher (kagglehub)
```

## Documentation

This README is intentionally concise. Please see `Documentation/main.pdf` for the full report (motivation, design choices, and references).