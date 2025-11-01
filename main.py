"""
Wikipedia Movies Retrieval System - Main 
"""

import pandas as pd
import sys
sys.path.append('Components')
from Tokenizer import Tokenizer
import Indexer

tk = Tokenizer(remove_stopwords=False)
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
    lambda row: tk.tokenize(str(row['title']) + ' ' + str(row['plot'])),
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

# Build inverted index using SPIMI (memory-resident)
print("\n[6/7] Building Inverted Index (SPIMI)...")
print("-"*80)

# Initialize indexer state
index_state = Indexer.init_memory(tk)

# Index all documents
for idx, row in all_movies.iterrows():
    title = str(row['title'])
    text = str(row['title']) + ' ' + str(row['plot'])
    Indexer.index_doc_mem(index_state, title, text)
    
    if (idx + 1) % 5000 == 0:
        print(f"  Indexed {idx + 1} documents...")

print(f"  ✓ Indexed {index_state['next_id']} documents")

# Index statistics
num_docs = index_state['next_id']
vocab_size = len(index_state['index'])
total_postings = sum(len(postings) for postings in index_state['index'].values())
avg_postings = total_postings / vocab_size if vocab_size > 0 else 0

print(f"\nIndex Statistics:")
print(f"  - Documents indexed: {num_docs:,}")
print(f"  - Vocabulary size: {vocab_size:,}")
print(f"  - Total postings: {total_postings:,}")
print(f"  - Avg postings per term: {avg_postings:.1f}")

# Show some example postings
print("\n[7/7] Example Index Entries:")
print("-"*80)
example_terms = ['love', 'kill', 'friend', 'family', 'death']
for term in example_terms:
    postings = Indexer.postings_mem(index_state, term)
    doc_freq = len(postings)
    print(f"\nTerm: '{term}'")
    print(f"  Document frequency: {doc_freq}")
    print(f"  Sample postings (doc_id: freq): {list(postings.items())[:5]}")

print("\n" + "="*80)
print("System initialized successfully!")
print(f"Ready to search {num_docs:,} movies with {vocab_size:,} unique terms")
print("="*80)

# Store index state globally for later use (querying, ranking)
all_movies.index_state = index_state

from QueryProcessor import QueryProcessor

print("\n[8/8] Testing Query Processor")
print("-"*80)

qp = QueryProcessor(index_state)

query = "space mission to Mars"

results = qp.rank_smart(query, weighting="ltc.lnc")

print(f"\nTop results for query: '{query}'")
for rank, (title, score) in enumerate(results, 1):
    print(f"{rank:2d}. {title:50s}  cosine_sim={score:.4f}")