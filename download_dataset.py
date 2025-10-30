import kagglehub
import os
import shutil

# Download dataset from Kaggle
print("Downloading dataset...")
cache_path = kagglehub.dataset_download("exactful/wikipedia-movies")

# Create data folder in repo
os.makedirs("data", exist_ok=True)

# Copy all CSV files to data folder
print(f"\nCopying files to ./data/")
for file in os.listdir(cache_path):
    if file.endswith('.csv'):
        src = os.path.join(cache_path, file)
        dst = os.path.join("data", file)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst) / (1024*1024)
        print(f"  ✓ {file} ({size:.1f} MB)")

print(f"\nDone! Files are in: {os.path.abspath('data')}")
