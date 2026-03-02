#!/usr/bin/env python3
"""Check actual VIX statistics from data."""
import pandas as pd
from pathlib import Path
import numpy as np

vix_path = Path('/mnt/d/Mamba v2/datasets/VIX')
all_closes = []
for f in sorted(vix_path.glob('VIX_*.csv')):
    df = pd.read_csv(f, usecols=['close'])
    all_closes.extend(df['close'].dropna().tolist())

arr = np.array(all_closes)
print(f'VIX samples: {len(arr)}')
print(f'Mean: {arr.mean():.2f}')
print(f'Std: {arr.std():.2f}')
print(f'Min: {arr.min():.2f}')
print(f'Max: {arr.max():.2f}')
print(f'Median: {np.median(arr):.2f}')
