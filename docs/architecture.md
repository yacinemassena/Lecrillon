# VIX Prediction Architecture

## System Requirements

**IMPORTANT: WSL (Windows Subsystem for Linux) Required for Training**

The Mamba SSM package requires CUDA compilation with Linux toolchain and cannot be built on native Windows. All training must be done in WSL or a Linux environment.

### Setup on Windows:
1. Install/repair WSL: `wsl --install -d Ubuntu` (PowerShell as Admin)
2. Run setup: `cd /mnt/d/Mamba\ v2 && bash setupenv.sh`
3. This creates `venv/` with custom `mamba_blackwell` (sm_120 Blackwell support)
4. All training commands must be run in WSL with this venv activated

### Why WSL is Required:
- Mamba SSM uses CUDA C++ extensions compiled via PyTorch's `CUDAExtension`
- Requires `nvcc` (CUDA compiler) + Linux build tools (`gcc`, `ninja`)
- Windows native Python cannot compile these CUDA kernels
- The `.so` files in `custom_packages/mamba_blackwell/` are Linux binaries

---

Final architecture summary
Level 0: Transformer Frame Encoder (compress 1s bars into 5-min frames)
Input: Stock 1-second bar data (top-100 tickers, 15 features per bar)
Group bars into 5-minute frames (~300 bars per frame, 78 frames per trading day)
BarEncoder: Transformer (6 layers, 256 dim, 4 heads) with masked mean pooling
Output per frame: [256] embedding
Trainable end-to-end (no pretraining)
Auxiliary Task: Predict 15-min forward Realized Volatility (RV) from this embedding.
Already-aggregated numeric features (fixed-size per 5min)
Use small MLP
Output per stream: e.g. [64]
Text / macro / fed
Use pretrained text embedding model (frozen) + small adapter MLP, then:
FiLM for slow macro regime vector
optionally a small news_state vector per 5min (pooled headlines)
Level 1: Mamba-1 (short-term dynamics)
Input: 15-day lookback of frame embeddings = 15 × 78 = 1,170 steps
Each step is a 5-min frame embedding [256] projected to d_model
Model: Mamba-1 (4–8 layers, d_model=256–512)
Output: next-day VIX close (single scalar)
Also produces daily summary embeddings (pool each day's Mamba-1 hidden states)
Level 2: Mamba-2 (long-term regime)
Input: 365-day lookback of daily summary embeddings = 365 steps
Each step is one day's pooled Mamba-1 output [d_model]
Model: Mamba-2 (4–8 layers, d_model=256–512)
Output: VIX +30d close (single scalar)
Training
End-to-end backprop through everything (Transformer encoder + Mamba-1 + Mamba-2), unless you choose to pretrain Transformer and later fine-tune.
Combined Loss: Main VIX Prediction + RV Prediction (Auxiliary)
Data structure (your streams), simplified and corrected
Input streams → 5min embeddings
5min frame encoders (run every 5min)
├─ Stock 1s bars (top-100 tickers)   ─→ [P, S, dt, TickerID] → Transformer(6L) + Pool ─→ e_stk   [256] ──(Aux)──→ RV Pred
├─ Option trades 1s                 ─→ [P, S, dt, TickerID] → Transformer + Pool      ─→ e_opt   [256]
├─ Quote aggregates (5min)          ─→ MLP(3)                 ─→ e_quote [64]
├─ Index bars (1m)                  ─→ MLP(2) + time-align    ─→ e_idx   [64]
└─ News/Macro/FED
      ├─ News embeddings (frozen) + pool per 5min → e_news [128] (or [64])
      └─ Macro/FED numeric vector → FiLM(gamma,beta) on Mamba hidden state
Then fuse:
e_5min = concat(e_stk, e_opt, e_quote, e_idx, e_news, calendar_feats)
e_5min → Linear/MLP projection → d_model (e.g., 512)
Sequence models
e_5min sequence (15 days ≈ 1,170 steps)
   → Mamba-1 (4–8 layers, d_model=256–512)
   → next-day VIX head
daily summary (from Mamba-1 outputs)
   → Mamba-2 (4–8 layers, d_model=256–512)
   → VIX +30d head
Mamba 2
Input streams (different frequencies):
├─ Stock 1s bars (top-100 tickers) ────→ [Features + TickerEmb] → Transformer(6L) + Pool ────→ [256]
├─ Option trades 1s ────→ [Features + TickerEmb] → Transformer + Pool ────→ [256]
├─ Quote aggregates (5min) ─→ MLP-3 layers ─────→ [64]
├─ Index minute bars ────────→ MLP-2 layers ────→ [64]
└─ News/Macro/FED ───────────→ BERT encoder ─────→ [32] FiLM
Walk forward validation (validate on data in the future of the batch)
Typical flow:
Use the 2y/1y setup to pick:
architecture (encoder/Mamba sizes, fusion scheme, FiLM design),
losses/weights for your 3 horizons,
regularization, optimizer, LR schedule,
sampling and sequence lengths.
Once decided, run full training on ~20 years.
TRAINING :
A common scheme:
Train: first ~17–18 years
Val: next ~1 year (for early stopping / model selection)
Test: last ~1 year (final performance)
And because you have 30-day sequences, keep a gap (~30 days + label horizon) between each split.
NEWS/EVENTS :
Fed Events → Event MemoryWhy: Discrete, timestamped, impact decays over time, future events matterpythonevent_memory = [
    {type: "FOMC_DECISION", time: -15d, rate: 5.50, decision: "hold", tone: "hawkish"},
    {type: "FED_MINUTES", time: -8d, surprise: "dovish"},
    {type: "FED_SPEAKER", time: -2d, who: "Powell", tone: "neutral"},
    {type: "FOMC_SCHEDULED", time: +5d},  # FUTURE - model knows it's coming
]Key insight: Markets price in future Fed meetings. The anticipation matters as much as the event itself. Event Memory can include upcoming scheduled events:pythonclass FedEventMemory(nn.Module):
    def encode_event(self, event):
        emb = self.type_embed(event.type)  # FOMC, minutes, speaker
        emb += self.time_embed(event.days_from_now)  # negative = past, positive = future
        emb += self.content_embed(event.details)  # rate, tone, surprise
        return embModel learns:
"FOMC in 2 days → vol elevated, positioning matters"
"FOMC was 10 days ago, hawkish → still affecting rate-sensitive flows"
"Fed speaker tomorrow → might wait on big positions"
Macro Stats → FiLMWhy: Slow-moving, continuous values, affects everything uniformly, no discrete eventspythonmacro_regime = {
    gdp_growth: 2.1,
    unemployment: 3.8,
    cpi_yoy: 3.4,
    pce_core: 2.8,
    yield_10y: 4.3,
    yield_2y: 4.7,
    yield_curve_10y2y: -0.4,  # inverted
    credit_spreads: 1.2,
    dollar_index: 104.5,
    oil_price: 78.0,
    vix: 14.5,
    economic_surprise_index: 0.3,
}This is your "economic regime lens." FiLM applies it globally:python# Model learns: "when yield curve inverted + unemployment low + CPI sticky,
# interpret options skew signals differently than in easing cycle"
gamma, beta = self.film_proj(macro_vector)
h = gamma * h + betaLet me break this down by data source and what mechanism fits each:
Fed Events → Event Memory
Why: Discrete, timestamped, impact decays over time, future events matter
pythonevent_memory = [
    {type: "FOMC_DECISION", time: -15d, rate: 5.50, decision: "hold", tone: "hawkish"},
    {type: "FED_MINUTES", time: -8d, surprise: "dovish"},
    {type: "FED_SPEAKER", time: -2d, who: "Powell", tone: "neutral"},
    {type: "FOMC_SCHEDULED", time: +5d},  # FUTURE - model knows it's coming
]
Key insight: Markets price in future Fed meetings. The anticipation matters as much as the event itself. Event Memory can include upcoming scheduled events:
pythonclass FedEventMemory(nn.Module):
    def encode_event(self, event):
        emb = self.type_embed(event.type)  # FOMC, minutes, speaker
        emb += self.time_embed(event.days_from_now)  # negative = past, positive = future
        emb += self.content_embed(event.details)  # rate, tone, surprise
        return emb
Model learns:
"FOMC in 2 days → vol elevated, positioning matters"
"FOMC was 10 days ago, hawkish → still affecting rate-sensitive flows"
"Fed speaker tomorrow → might wait on big positions"
Macro Stats → FiLM
Why: Slow-moving, continuous values, affects everything uniformly, no discrete events
pythonmacro_regime = {
    gdp_growth: 2.1,
    unemployment: 3.8,
    cpi_yoy: 3.4,
    pce_core: 2.8,
    yield_10y: 4.3,
    yield_2y: 4.7,
    yield_curve_10y2y: -0.4,  # inverted
    credit_spreads: 1.2,
    dollar_index: 104.5,
    oil_price: 78.0,
    vix: 14.5,
    economic_surprise_index: 0.3,
}
This is your "economic regime lens." FiLM applies it globally:
python# Model learns: "when yield curve inverted + unemployment low + CPI sticky,
# interpret options skew signals differently than in easing cycle"
gamma, beta = self.film_proj(macro_vector)
h = gamma * h + beta
Don't use Event Memory for this because:
No discrete events to retrieve
Values change slowly (monthly releases)
Impact is ambient, not time-decaying
Benzinga Real-time News → Cross-Attention + Recency Weighting
Why: Continuous stream, relevance varies per tick, need to select what matters
pythonnews_buffer = [
    {time: -30s, text_emb: [...], ticker: "NVDA", sentiment: 0.8, category: "earnings"},
    {time: -45s, text_emb: [...], ticker: "AAPL", sentiment: -0.2, category: "analyst"},
    {time: -2min, text_emb: [...], ticker: None, sentiment: 0.1, category: "sector"},
    ... last N headlines
]
Cross-attention lets each tick decide what's relevant:
pythonclass NewsAttention(nn.Module):
    def forward(self, hf_states, news_buffer):
        # Add recency decay to attention scores
        recency_weights = exp(-lambda * seconds_ago)
       
        attn_scores = einsum(hf_states, news_embs, "b t d, b n d -> b t n")
        attn_scores = attn_scores + log(recency_weights)  # bias toward recent
       
        return attention_weighted_sum(attn_scores, news_embs)
Why not FiLM: News is heterogeneous. A tick processing NVDA options should attend to NVDA news, not AAPL news. FiLM would blend everything.
Why not pure Event Memory: News isn't discrete events, it's a continuous stream. You want soft attention over recent headlines, not retrieval of specific past events.
GDELT : filter by CAMEO codes + goldstein
GDELT Pipeline Summary
Step 1: Filter (before any compute)
python# Keep if:
goldstein_scale < -5          # conflict/negative events
OR cameo_root in [13,14,16,17,18,19,20]  # threaten, protest, coerce, assault, fight, mass violence
OR cameo_code in ['166','167','173']      # sanctions, embargo, blockade specifically
# Also filter by:
actor_country in relevant_countries  # US, CN, RU, IR, SA, TW, etc.
50,000 articles → ~200-500 relevant
Step 2: Dedupe + Score
python# Remove duplicate stories (same event, different sources)
# Score by: goldstein severity × num_sources × actor importance
# Keep top 20
Step 3: Encode (MVP - no text)
pythonfeatures = [
    goldstein_scale,
    num_sources,
    num_mentions,
    cameo_root_onehot,      # 20 dims
    actor1_country_onehot,  # 15 dims
    actor2_country_onehot,  # 15 dims
    region_onehot,          # 10 dims
]
# ~65 features → small MLP → embedding
Step 4: Two outputs
Bad stuff → Event Memory (with history)
pythonif goldstein < -7 or is_anomaly:
    event_memory.append(event)  # keep 15 days
# → enters model via cross-attention
# → model can retrieve "what happened 3 days ago"
Everything relevant → World State (no history)
pythonworld_state = GRU(encoded_articles, world_state)  # accumulates implicitly
# → enters model via FiLM
# → model sees "current geopolitical regime"
```
### Architecture
```
GDELT (every 15min)
    │
    ├── Filter (Goldstein + CAMEO + Countries)
    │
    ├── Encode (structured features, no text)
    │
    ├──→ Severe events ──→ Event Memory (15 days) ──→ Cross-Attention
    │
    └──→ All relevant ──→ GRU ──→ World State ──→ FiLM
Model integration
pythonh = Mamba(tick_data)
h = gdelt_film(world_state) * h + beta      # regime context
h = h + attention(h, event_memory)           # specific event lookup