# WikipediaMoviesRetrieval

Information Retrieval project using Wikipedia Movies dataset from Kaggle.

## Setup

### 1. Install Dependencies

Install the Kaggle API package:

```bash
pip install kaggle
```

### 2. Configure Kaggle Credentials

You need a Kaggle API token to download datasets. Follow these steps:

1. Go to [kaggle.com](https://www.kaggle.com) and log in
2. Click on your profile picture → **Settings**
3. Scroll down to **API** section
4. Click **Create New Token** (downloads `kaggle.json`)
5. Move the file to the correct location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Alternatively, you can set environment variables:

```bash
export KAGGLE_USERNAME='your_username'
export KAGGLE_KEY='your_api_key'
```

### 3. Download the Dataset

Run the download script:

```bash
python download_dataset.py
```

The dataset will be downloaded to `data/wikipedia-movies/` directory.

## Dataset

- **Source**: [exactful/wikipedia-movies on Kaggle](https://www.kaggle.com/datasets/exactful/wikipedia-movies)
- **Content**: Wikipedia data about movies