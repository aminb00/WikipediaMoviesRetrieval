"""
Download Wikipedia Movies dataset from Kaggle using kagglehub.
"""
import kagglehub
import os


def download_wikipedia_movies_dataset():
    """Download the latest version of the Wikipedia Movies dataset."""
    print("Downloading 'exactful/wikipedia-movies' dataset from Kaggle...")
    
    # Download latest version
    path = kagglehub.dataset_download("exactful/wikipedia-movies")
    
    print(f"\nDataset path: {path}")
    
    # List files
    if os.path.exists(path):
        files = os.listdir(path)
        if files:
            for f in sorted(files):
                filepath = os.path.join(path, f)
                size = os.path.getsize(filepath) / (1024*1024)  # MB
                print(f"  - {f} ({size:.1f} MB)")
        else:
            parent = os.path.dirname(path)
            if os.path.exists(parent):
                all_files = []
                for root, dirs, filenames in os.walk(parent):
                    for filename in filenames:
                        if filename.endswith('.csv'):
                            full_path = os.path.join(root, filename)
                            all_files.append(full_path)
                            size = os.path.getsize(full_path) / (1024*1024)
                            print(f"  - {full_path} ({size:.1f} MB)")
                
                if all_files:
                    return all_files[0].rsplit('/', 1)[0]  # Return directory
    
    return path


if __name__ == "__main__":
    dataset_path = download_wikipedia_movies_dataset()
    print(f"\nDataset path: {dataset_path}")
