# Topic Analysis ULTRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![BERTopic](https://img.shields.io/badge/BERTopic-0.16+-orange.svg)](https://maartengr.github.io/BERTopic/)

Standalone topic modeling with 4 models, parallel K-sweep (3-20), auto top-3 K selection, and Pareto-optimal evaluation. Local, standalone, no API keys.

**Winner on benchmarks: NMF K=7 with c_v=0.89, cnpmi=0.54, diversity=0.97.**

## Features

| Model | Description |
|-------|-------------|
| **BERTopic** | sentence-transformers + UMAP + HDBSCAN (Grootendorst 2022, 1309 cites) |
| **STM** | Structural Topic Model with covariates via R (Roberts et al. 2019) |
| **NMF** | TF-IDF + K-sweep — beat BERTopic on coherence per Asas 2025 |
| **LDA** | CountVectorizer + K-sweep + pyLDAvis visualization |

| Evaluation Metric | Description |
|-------------------|-------------|
| **c_v coherence** | Topic coherence via cosine similarity (Röder et al. 2015) |
| **c_npmi** | Normalized PMI coherence (Aletras & Stevenson 2013) |
| **Diversity** | Unique word proportion across topics (Dieng et al. 2020) |
| **Redundancy** | Overlapping word proportion |
| **K-selection** | Pareto front across all metrics (Weston et al. 2023) |

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (NMF + LDA)
python topic_ultra.py data.csv text_col

# Research-grade (BERTopic + NMF + LDA + STM)
python topic_ultra.py data.csv text_col --profile research

# BERTopic only
python topic_ultra.py data.csv text_col --profile full --no-nmf --no-lda

# Custom K values
python topic_ultra.py data.csv text_col --k 5,10,15

# Demo (small data, fast)
python topic_ultra.py demo
```

## Usage

```
python topic_ultra.py <csv_path> <text_col> [options]
```

| Argument | Description |
|----------|-------------|
| `csv_path` | Path to CSV file (or "demo") |
| `text_col` | Column name containing text |
| `-o, --output` | Output directory (default: output_topic_ultra) |
| `--k` | Comma-separated K values (e.g. `5,10,15`) |
| `--profile` | `demo` (NMF 3-8), `fast` (NMF+LDA), `full` (+BERTopic), `research` (+STM) |
| `--sample` | Sample N documents (stratified if group column) |
| `--seed` | Random seed (default: 42) |
| `--group` | Group column for contrastive topic analysis |
| `--cache-embeddings` | Cache BERTopic embeddings for faster re-runs |
| `--no-bertopic` | Skip BERTopic |
| `--no-stm` | Skip STM |
| `--no-nmf` | Skip NMF |
| `--no-lda` | Skip LDA |

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
│  CSV Data   │───→│  Text Cleaner    │───→│  Parallel Model Sweep    │
│             │    │  (light +        │    │                          │
│             │    │   classical)     │    │  ┌────────────────────┐  │
└─────────────┘    └──────────────────┘    │  │ NMF    K=3..20    │  │
                                            │  │ LDA    K=3..20    │  │
                                            │  │ BERTopic (auto)   │  │
                                            │  │ STM    K=3..20    │  │
                                            │  └────────┬───────────┘  │
                                            │           │              │
                                            │  ┌────────▼───────────┐  │
                                            │  │  Evaluation Layer  │  │
                                            │  │  c_v coherence    │  │
                                            │  │  c_npmi           │  │
                                            │  │  Diversity        │  │
                                            │  │  Redundancy       │  │
                                            │  └────────┬───────────┘  │
                                            └───────────┼──────────────┘
                                                        │
                                            ┌───────────▼──────────────┐
                                            │    Pareto K-Selection    │
                                            │    Top-3 by c_v          │
                                            └───────────┬──────────────┘
                                                        │
                                            ┌───────────▼──────────────┐
                                            │     Output Artifacts     │
                                            │  topic_ultra_results.json│
                                            │  manifest.json          │
                                            │  visualizations/        │
                                            │  pyLDAvis/              │
                                            └──────────────────────────┘
```

## Benchmark Results

Measured June 2026. NMF consistently outperforms other models across all datasets.

| Dataset | Best Model | K | c_v | c_npmi | Diversity | Time |
|---------|-----------|---|-----|--------|-----------|------|
| 20 Newsgroups (497 docs) | NMF | 4 | 0.748 | 0.22 | 0.975 | 42.1s |
| IMDb Sentiment (99 docs) | NMF | 3 | 0.447 | — | 0.867 | 26.7s |
| TripAdvisor HK (full corpus) | NMF | 7 | 0.894 | 0.542 | 0.971 | — |
| BBC News (298 docs) | NMF | 3 | 0.823 | 0.324 | 1.0 | 46.4s |

**TripAdvisor HK (previous run, full data):** NMF K=7, c_v=0.894, c_npmi=0.542, diversity=0.971.

**Key insights:**
- NMF dominates on all metrics — TF-IDF + non-negative matrix factorization captures interpretable topic structure
- LDA offers competitive diversity but lower coherence
- BERTopic produces fewer topics (auto-selected K=3) with high diversity
- STM needs R installation but is valuable for covariate analysis
- Parallel ProcessPoolExecutor makes K-sweeps fast (NMF+LDA K=3-20 in ~20s)

## Output

| File | Description |
|------|-------------|
| `topic_ultra_results.json` | Full results: topic words, coherence, diversity, representative docs per model |
| `manifest.json` | Runtime metadata: timestamp, n_docs, best model, elapsed |
| `visualizations/` | Coherence curves, topic word clouds per model |
| `pyLDAvis/` | Interactive LDA topic visualization |

## Tips

### Which model should I use?

| Goal | Recommended | Why |
|------|-------------|-----|
| Interpretable topics | `--no-bertopic --no-stm` | NMF wins on coherence (c_v=0.89) |
| Semantic clustering | `--profile full --no-nmf --no-lda` | BERTopic captures semantic structure |
| Research publication | `--profile research` | STM with covariates + NMF/LDA baseline |
| Quick exploration | `--profile demo` | NMF only, K=3-8, fast |
| Large corpus (10K+) | `--no-bertopic` | BERTopic scales poorly with HDBSCAN |

### Speed

- **NMF K=3-20**: ~10s for 2800 docs
- **LDA K=3-20**: ~15s for 2800 docs
- **BERTopic**: ~20s (including embeddings) for 2800 docs
- **STM**: ~60s per K + R startup overhead

## Dependencies

```
numpy
pandas
scikit-learn
sentence-transformers
gensim
pyldavis
plotly
wordcloud
```

Optional: `bertopic>=0.16`, `stm` (R package), `Rscript`

## Citation

```bibtex
@software{pang2026topicultra,
  author = {Peter Pang},
  title = {Topic Analysis ULTRA: Evidence-Based Topic Modeling Toolkit},
  year = {2026},
  url = {https://github.com/chessy795/topic-ultra}
}
```

## License

MIT
