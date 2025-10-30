"""
Wikipedia Movies Retrieval System - Main 
"""

import pandas as pd
import sys
sys.path.append('Components')
from Tokenizer import tokenize
from Indexer import build_index

print("="*80)
print("Wikipedia Movies Retrieval System")
print("="*80)

# Load all movie datasets
print("\n[1/5] Loading datasets...")
dataframes = []
for decade in ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']:
    df = pd.read_csv(f'data/{decade}-movies.csv')
    df['decade'] = decade
    dataframes.append(df)
    print(f"  ✓ Loaded {len(df)} movies from {decade}")

# Combine all movies
all_movies = pd.concat(dataframes, ignore_index=True)
print(f"\nTotal movies loaded: {len(all_movies)}")

# Data exploration
print("\n[2/5] Data Exploration...")
print("-"*80)
print("\nDataset Info:")
print(f"  - Columns: {list(all_movies.columns)}")
print(f"  - Shape: {all_movies.shape}")
print(f"  - Missing values: {all_movies.isnull().sum().to_dict()}")

print("\n\nFirst 3 movies:")
print("-"*80)
for i in range(3):
    movie = all_movies.iloc[i]
    print(f"\n{i+1}. {movie['title']} ({movie['decade']})")
    plot_preview = movie['plot'][:150] + "..." if len(movie['plot']) > 150 else movie['plot']
    print(f"   Plot: {plot_preview}")

print("\n\nMovies per decade:")
print(all_movies['decade'].value_counts().sort_index())

print("\n\nPlot length statistics:")
all_movies['plot_length'] = all_movies['plot'].str.len()
print(f"  - Mean: {all_movies['plot_length'].mean():.0f} characters")
print(f"  - Median: {all_movies['plot_length'].median():.0f} characters")
print(f"  - Min: {all_movies['plot_length'].min():.0f} characters")
print(f"  - Max: {all_movies['plot_length'].max():.0f} characters")

# Tokenize all documents
print("\n[3/5] Tokenizing documents...")
print("Processing movie titles and plots...")

# Tokenize each movie (title + plot)
all_movies['tokens'] = all_movies.apply(
    lambda row: tokenize(str(row['title']) + ' ' + str(row['plot'])),
    axis=1
)

# Count total tokens
total_tokens = sum(len(tokens) for tokens in all_movies['tokens'])
print(f"  ✓ Processed {len(all_movies)} documents")
print(f"  ✓ Total tokens: {total_tokens:,}")

# Tokenization analysis
print("\n[4/5] Tokenization Analysis...")
print("-"*80)

# Token count per document
all_movies['token_count'] = all_movies['tokens'].apply(len)
print(f"\nTokens per document statistics:")
print(f"  - Mean: {all_movies['token_count'].mean():.1f} tokens")
print(f"  - Median: {all_movies['token_count'].median():.1f} tokens")
print(f"  - Min: {all_movies['token_count'].min()} tokens")
print(f"  - Max: {all_movies['token_count'].max()} tokens")

# Build vocabulary
print(f"\nBuilding vocabulary...")
vocabulary = set()
for tokens in all_movies['tokens']:
    vocabulary.update(tokens)
print(f"  ✓ Unique tokens in vocabulary: {len(vocabulary):,}")

# Most common tokens
from collections import Counter
all_tokens_flat = [token for tokens in all_movies['tokens'] for token in tokens]
token_freq = Counter(all_tokens_flat)
print(f"\nTop 20 most frequent tokens:")
for token, count in token_freq.most_common(20):
    print(f"  {token:20s} : {count:6,} occurrences")

# Show sample
print("\n[5/5] Sample tokenized documents:")
print("-"*80)
for i in range(3):
    movie = all_movies.iloc[i]
    print(f"\n{i+1}. {movie['title']} ({movie['decade']})")
    print(f"   Original plot length: {len(movie['plot'])} chars")
    print(f"   Tokens ({len(movie['tokens'])}): {movie['tokens'][:15]}...")

# Build inverted index
print("\n[6/7] Building Inverted Index...")
print("-"*80)
index = build_index(all_movies)

# Index statistics
stats = index.get_stats()
print(f"\nIndex Statistics:")
print(f"  - Documents indexed: {stats['num_documents']:,}")
print(f"  - Vocabulary size: {stats['vocabulary_size']:,}")
print(f"  - Total postings: {stats['total_postings']:,}")
print(f"  - Avg postings per term: {stats['avg_postings_per_term']:.1f}")

# Show some example postings
print("\n[7/7] Example Index Entries:")
print("-"*80)
example_terms = ['love', 'kill', 'friend', 'family', 'death']
for term in example_terms:
    postings = index.get_postings(term)
    doc_freq = index.get_document_frequency(term)
    print(f"\nTerm: '{term}'")
    print(f"  Document frequency: {doc_freq}")
    print(f"  Sample postings (doc_id, freq): {postings[:5]}")

print("\n" + "="*80)
print("System initialized successfully!")
print(f"Ready to search {len(all_movies):,} movies with {len(vocabulary):,} unique terms")
print("="*80)
