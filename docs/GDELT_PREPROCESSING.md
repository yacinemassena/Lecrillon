# GDELT Preprocessing for VIX Prediction

**Goal:** Transform raw GDELT data (thousands of articles per 15-min bucket) into single embedding tokens that capture "world state" for integration into a Mamba sequence model predicting VIX.

---

## Context

### What is GDELT?
- Global Database of Events, Language, and Tone
- Monitors news worldwide, extracts structured events
- Updated every 15 minutes
- ~1000-2000 articles per 15-min bucket
- Each record includes: title, URL, Goldstein scale (-10 to +10), CAMEO event codes, actors, locations, tone, sources

### How It's Used
The Mamba model processes 1-second stock bars (~23,400/day). News is **interleaved** as special tokens:

```
bars[0-899] → GDELT_09:30 → bars[900-1799] → GDELT_09:45 → bars[1800-2699] → ...
```

Each GDELT token is inserted at the end of its 15-min window. Mamba processes causally — bars can only see PAST GDELT tokens.

**Result:** ~78 GDELT tokens per trading day (6.5 hours × 4 buckets/hour).

---

## Output Format

Each 15-min bucket → **one vector** saved to parquet:

```
datasets/GDELT/
├── 2024/
│   ├── 01/
│   │   ├── 02.parquet   # All buckets for Jan 2, 2024
│   │   ├── 03.parquet
│   │   └── ...
```

**Parquet schema:**
```python
{
    'bucket_start': datetime,      # e.g., 2024-01-02 09:30:00 UTC
    'bucket_end': datetime,        # e.g., 2024-01-02 09:44:59 UTC
    'embedding': np.ndarray,       # [D] float32 — the aggregate embedding
    'article_count': int,          # number of articles in bucket
    'min_goldstein': float,        # worst event severity
    'mean_goldstein': float,       # average severity
    'max_sources': int,            # most-covered story
    'conflict_ratio': float,       # % military/conflict CAMEO codes
    'protest_ratio': float,        # % protest codes
    'economic_ratio': float,       # % economic-related codes
}
```

---

## Embedding Dimension Options

### Option A: 64 dims (Recommended Start)

```python
pooled_embedding = [64]  # PCA-reduced from 384
summary_stats = [8]       # min/mean goldstein, article_count, etc.
# Total: 72 dims
```

**Pros:** Compact, fast to train, sufficient for "world sentiment" signal.
**Cons:** May lose nuance distinguishing event types.

### Option B: 128 dims

```python
pooled_embedding = [128]  # PCA from 384 or 768
summary_stats = [8]
# Total: 136 dims
```

**Pros:** Better semantic resolution.
**Cons:** Slightly more memory.

### Option C: 384 dims (Full MiniLM)

```python
pooled_embedding = [384]  # No PCA, raw MiniLM
summary_stats = [8]
# Total: 392 dims
```

**Pros:** Maximum semantic information.
**Cons:** Larger storage, projection layer does more compression.

### Recommendation

**Start with 64+8=72 dims.** The model's `d_model` (256-512) is the bottleneck anyway — a `Linear(72, 256)` projection can extract what it needs. Upgrade to 128 if ablations show benefit.

---

## Preprocessing Pipeline

### Step 1: Download GDELT Data

GDELT provides 15-min update files:

```python
# GDELT 2.0 URLs (updated every 15 min)
base_url = "http://data.gdeltproject.org/gdeltv2/"
# Files like: 20240102091500.gkg.csv.zip

# Or use BigQuery (faster for bulk):
from google.cloud import bigquery
client = bigquery.Client()

query = """
SELECT 
    GKGRECORDID, DATE, SourceCommonName, DocumentIdentifier,
    V2Tone, V2Themes, V2Locations, V2Persons, V2Organizations
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE DATE >= '2024-01-01' AND DATE < '2024-01-02'
"""
```

### Step 2: Filter Relevant Articles

Not all GDELT records are useful. Filter to:

```python
def filter_gdelt(df):
    """Keep only market-relevant articles."""
    
    # Option A: By Goldstein scale (negative = conflict/instability)
    severe = df[df['goldstein_scale'] < -3]
    
    # Option B: By CAMEO root codes
    relevant_codes = [
        '14',  # Protest
        '17',  # Coerce
        '18',  # Assault
        '19',  # Fight
        '20',  # Use unconventional mass violence
        '01',  # Make public statement (central bank, govt)
        '04',  # Consult (diplomatic)
    ]
    relevant = df[df['cameo_root'].isin(relevant_codes)]
    
    # Option C: By keywords in themes
    economic_keywords = ['ECON_', 'TAX_', 'CENTRAL_BANK', 'INFLATION', 'TRADE']
    has_econ = df['themes'].str.contains('|'.join(economic_keywords), na=False)
    
    return df[severe | relevant | has_econ]
```

**Target:** ~50-200 articles per 15-min bucket after filtering (vs 1000-2000 raw).

### Step 3: Embed Article Titles

Use a sentence transformer for embeddings:

```python
from sentence_transformers import SentenceTransformer
import torch

# Models ranked by quality/speed tradeoff:
# - all-MiniLM-L6-v2: 384 dims, 14k sent/sec (RECOMMENDED)
# - all-mpnet-base-v2: 768 dims, 2.5k sent/sec
# - paraphrase-multilingual-MiniLM-L12-v2: 384 dims, handles non-English

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_articles(titles: list[str]) -> np.ndarray:
    """Embed article titles. Returns [N, 384]."""
    embeddings = model.encode(
        titles, 
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings  # [N, 384]
```

### Step 4: Aggregate to Single Embedding

Pool all article embeddings into one vector per bucket:

```python
import numpy as np
from scipy.special import softmax

def aggregate_bucket(articles_df, embeddings: np.ndarray, pca=None) -> dict:
    """
    Aggregate N articles into one bucket embedding.
    
    Args:
        articles_df: DataFrame with goldstein_scale, num_sources, cameo_root, etc.
        embeddings: [N, 384] article embeddings
        pca: Optional fitted PCA to reduce dims (384 → 64)
    
    Returns:
        dict with 'embedding' [D] and summary stats
    """
    N = len(articles_df)
    
    if N == 0:
        # Empty bucket — return zeros
        embed_dim = 64 if pca else 384
        return {
            'embedding': np.zeros(embed_dim, dtype=np.float32),
            'article_count': 0,
            'min_goldstein': 0.0,
            'mean_goldstein': 0.0,
            'max_sources': 0,
            'conflict_ratio': 0.0,
            'protest_ratio': 0.0,
            'economic_ratio': 0.0,
        }
    
    # Weighted mean by severity (more negative goldstein = higher weight)
    goldstein = articles_df['goldstein_scale'].values
    weights = softmax(-goldstein)  # negative goldstein → high weight
    
    pooled = np.sum(weights[:, None] * embeddings, axis=0)  # [384]
    
    # Optional: reduce to 64 dims
    if pca:
        pooled = pca.transform(pooled.reshape(1, -1))[0]  # [64]
    
    # Summary statistics
    cameo = articles_df['cameo_root'].values
    return {
        'embedding': pooled.astype(np.float32),
        'article_count': N,
        'min_goldstein': float(goldstein.min()),
        'mean_goldstein': float(goldstein.mean()),
        'max_sources': int(articles_df['num_sources'].max()),
        'conflict_ratio': float(np.isin(cameo, ['17','18','19','20']).mean()),
        'protest_ratio': float((cameo == '14').mean()),
        'economic_ratio': float(np.isin(cameo, ['01','04']).mean()),
    }
```

### Step 5: Fit PCA (Optional)

If using PCA compression, fit on a sample first:

```python
from sklearn.decomposition import PCA
import pickle

# Collect embeddings from ~1 month of data
all_embeddings = []  # Collect during preprocessing
for day in days:
    for bucket in buckets:
        all_embeddings.extend(bucket_embeddings)

all_embeddings = np.vstack(all_embeddings)  # [~100k, 384]

# Fit PCA
pca = PCA(n_components=64)
pca.fit(all_embeddings)

print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
# Typically 85-95% with 64 components

# Save PCA model
with open('gdelt_pca_64.pkl', 'wb') as f:
    pickle.dump(pca, f)
```

### Step 6: Save to Parquet

```python
import pyarrow as pa
import pyarrow.parquet as pq

def save_day(date: str, buckets: list[dict], output_dir: Path):
    """Save all buckets for one day to parquet."""
    
    # Convert embeddings to list for parquet
    records = []
    for b in buckets:
        records.append({
            'bucket_start': b['bucket_start'],
            'bucket_end': b['bucket_end'],
            'embedding': b['embedding'].tolist(),  # [D] → list
            'article_count': b['article_count'],
            'min_goldstein': b['min_goldstein'],
            'mean_goldstein': b['mean_goldstein'],
            'max_sources': b['max_sources'],
            'conflict_ratio': b['conflict_ratio'],
            'protest_ratio': b['protest_ratio'],
            'economic_ratio': b['economic_ratio'],
        })
    
    df = pd.DataFrame(records)
    
    # Save
    year, month, day = date.split('-')
    path = output_dir / year / month / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
```

---

## Full Pipeline Script Skeleton

```python
#!/usr/bin/env python
"""
GDELT Preprocessing Pipeline

Transforms raw GDELT data into aggregated 15-min bucket embeddings.

Usage:
    python preprocess_gdelt.py --start-date 2020-01-01 --end-date 2024-12-31
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--output-dir', default='datasets/GDELT')
    parser.add_argument('--embed-dim', type=int, default=64, choices=[64, 128, 384])
    parser.add_argument('--pca-model', default=None, help='Path to fitted PCA model')
    args = parser.parse_args()
    
    # Load models
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    pca = None
    if args.embed_dim < 384:
        if args.pca_model and Path(args.pca_model).exists():
            with open(args.pca_model, 'rb') as f:
                pca = pickle.load(f)
        else:
            print("WARNING: No PCA model provided, will fit on first month")
            # TODO: Implement PCA fitting
    
    # Process each day
    current = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        print(f"Processing {date_str}...")
        
        # 1. Download/load raw GDELT for this day
        raw_df = load_gdelt_day(date_str)  # TODO: implement
        
        # 2. Filter to relevant articles
        filtered_df = filter_gdelt(raw_df)
        
        # 3. Group by 15-min buckets
        buckets = []
        for bucket_start, bucket_df in filtered_df.groupby(pd.Grouper(key='timestamp', freq='15min')):
            # 4. Embed titles
            titles = bucket_df['title'].tolist()
            embeddings = embed_model.encode(titles) if titles else np.zeros((0, 384))
            
            # 5. Aggregate
            agg = aggregate_bucket(bucket_df, embeddings, pca)
            agg['bucket_start'] = bucket_start
            agg['bucket_end'] = bucket_start + timedelta(minutes=15)
            buckets.append(agg)
        
        # 6. Save
        save_day(date_str, buckets, Path(args.output_dir))
        
        current += timedelta(days=1)

if __name__ == '__main__':
    main()
```

---

## Loading in DataLoader

```python
def _load_gdelt(self, date: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load GDELT embeddings for a date. Returns (embeddings, timestamps)."""
    
    year, month, day = date.split('-')
    path = self.gdelt_dir / year / month / f"{day}.parquet"
    
    if not path.exists():
        # Return empty if no GDELT data
        return torch.zeros(0, 72), torch.zeros(0)
    
    df = pd.read_parquet(path)
    
    # Stack embeddings + stats
    embeddings = np.stack(df['embedding'].values)  # [N_buckets, 64]
    stats = df[['min_goldstein', 'mean_goldstein', 'max_sources', 
                'conflict_ratio', 'protest_ratio', 'economic_ratio',
                'article_count']].values  # [N_buckets, 7]
    stats[:, 6] = np.log1p(stats[:, 6])  # log-transform article count
    
    combined = np.concatenate([embeddings, stats], axis=1)  # [N_buckets, 71]
    
    # Convert timestamps to seconds since market open
    market_open = pd.Timestamp(f"{date} 09:30:00", tz='America/New_York')
    timestamps = (df['bucket_end'] - market_open).dt.total_seconds().values
    
    return torch.tensor(combined, dtype=torch.float32), torch.tensor(timestamps)
```

---

## Summary

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **Embedding model** | `all-MiniLM-L6-v2` | Fast, 384-dim, good quality |
| **PCA reduction** | 384 → 64 | ~90% variance retained |
| **Final embedding dim** | 72 (64 embed + 8 stats) | Projected to d_model in model |
| **Articles per bucket** | ~50-200 after filtering | From 1000-2000 raw |
| **Buckets per day** | 26 (6.5h × 4/hour) | Trading hours only |
| **Storage** | ~2KB/day | 26 buckets × 72 floats × 4 bytes |

---

## Questions to Resolve

1. **Timezone handling:** GDELT timestamps are UTC. Need to convert to ET market hours.
2. **Overnight news:** Include pre-market bucket (04:00-09:30 ET) as single token?
3. **Weekend aggregation:** Roll up weekend news into Monday open token?
4. **CAMEO code mapping:** Need full mapping of relevant codes to filter.
