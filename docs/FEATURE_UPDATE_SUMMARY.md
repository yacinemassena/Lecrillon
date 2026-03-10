# Feature Count Update Summary

## Issue
The model was only using **15 stock features** and **15 option features**, but the actual data contains:
- **Stock data**: 29 features (excluding `ticker`, `bar_timestamp`)
- **Options data**: 38 features (excluding `underlying`, `bar_timestamp`)

## Changes Made

### 1. Updated Feature Lists (`loader/bar_mamba_dataset.py`)

**Stock Features (15 → 29):**
```python
DEFAULT_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'trade_count',
    'price_mean', 'price_std', 'price_range', 'price_range_pct', 'vwap',
    'avg_trade_size', 'std_trade_size', 'max_trade_size', 'amihud',
    'buy_volume', 'sell_volume', 'signed_trade_count',
    'tick_arrival_rate', 'large_trade_ratio', 'inter_trade_std', 'tick_burst',
    'rv_intrabar', 'bpv_intrabar', 'price_skew',
    'close_vs_vwap', 'high_vs_vwap', 'low_vs_vwap', 'ofi',
]
```

**Option Features (15 → 38):**
```python
OPTION_FEATURES = [
    'call_volume', 'put_volume', 'call_trade_count', 'put_trade_count',
    'put_call_ratio_volume', 'put_call_ratio_count',
    'call_premium_total', 'put_premium_total',
    'near_volume', 'mid_volume', 'far_volume',
    'near_pc_ratio', 'far_pc_ratio', 'term_skew',
    'otm_put_volume', 'atm_volume', 'otm_call_volume',
    'skew_proxy', 'atm_concentration', 'deep_otm_put_volume',
    'total_large_trade_count', 'call_large_count', 'put_large_count',
    'net_large_flow', 'large_premium_total', 'sweep_intensity',
    'max_volume_surprise', 'avg_volume_surprise',
    'uoa_call_count', 'uoa_put_count',
    'total_volume', 'total_trade_count',
    'unique_contracts', 'unique_strikes', 'unique_expiries',
    'pc_ratio_vs_20d', 'call_volume_vs_20d', 'put_volume_vs_20d',
]
```

### 2. Updated Model Defaults

**`mamba_only_model.py`:**
- `MambaOnlyVIX.__init__`: `num_features: int = 15` → `num_features: int = 29`
- `ParallelMambaVIX.__init__`: `num_features: int = 15` → `num_features: int = 29`

**`train.py`:**
- `SyntheticBarDataset.__init__`: `num_features: int = 15` → `num_features: int = 29`
- Training setup: `num_features = 15` → `num_features = 29  # All stock features from Stock_Data_1s`

## Impact

### Model Capacity
- **Stock encoder input**: 15 → 29 features (+93%)
- **Options encoder input**: 15 → 38 features (+153%)
- More information available for VIX prediction

### New Features Added

**Stock (14 new features):**
- OHLC: `open`, `high`, `low`
- Price stats: `price_mean`, `price_range`
- Trade size: `std_trade_size`, `max_trade_size`
- Microstructure: `signed_trade_count`, `inter_trade_std`, `bpv_intrabar`, `price_skew`
- VWAP relationships: `close_vs_vwap`, `high_vs_vwap`, `low_vs_vwap`

**Options (23 new features):**
- Volume breakdown: `call_volume`, `put_volume`, `call_trade_count`, `put_trade_count`
- Premiums: `call_premium_total`, `put_premium_total`
- Term structure: `near_volume`, `mid_volume`, `far_volume`
- Moneyness: `otm_put_volume`, `atm_volume`, `otm_call_volume`, `deep_otm_put_volume`
- Large trades: `total_large_trade_count`, `call_large_count`, `put_large_count`, `large_premium_total`
- Unusual activity: `uoa_call_count`, `uoa_put_count`
- Surprises: `max_volume_surprise`, `avg_volume_surprise`
- Metadata: `unique_contracts`, `unique_strikes`, `unique_expiries`
- Ratios: `put_call_ratio_count`

## Verification

Run `python verify_features.py` to confirm:
```
Stock features: 29
Option features: 38
```

## Next Steps

The model will now use all available features from the data. This should improve prediction quality by providing:
1. More complete price action information (OHLC vs just close)
2. Better microstructure signals (trade size distribution, signed flow)
3. Richer options flow data (term structure, moneyness, unusual activity)
4. Historical context (vs_20d features)
