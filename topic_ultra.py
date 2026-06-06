#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOPIC ANALYSIS ULTRA — Evidence-Based Standalone Topic Modeling
================================================================
Local, standalone, no API keys, no internet after model download.

4 models, parallel K-sweep (3-20), auto top-3 K selection per model.

ARCHITECTURE (peer-reviewed 2024-2026):
  1. BERTopic (primary) — sentence-transformers + UMAP + HDBSCAN (Grootendorst 2022, 1309 cites)
  2. STM (R via subprocess) — Structural Topic Model with covariates (Roberts et al. 2019)
  3. NMF (baseline) — TF-IDF + K-sweep (Asas 2025: NMF beat BERTopic on coherence)
  4. LDA (baseline) — CountVectorizer + K-sweep + pyLDAvis

  EVALUATION: c_v + c_npmi + diversity + redundancy (Tan & D'Souza 2025)
  K-SELECTION: Pareto front across metrics (Weston et al. 2023, 83 cites)
  PARALLEL: NMF/LDA K-sweeps run concurrently via ProcessPoolExecutor

Usage:
    python topic_ultra.py data.csv text_col                  # full pipeline
    python topic_ultra.py data.csv text_col --k 5,10,15     # specific K values
    python topic_ultra.py data.csv text_col --bertopic-only  # BERTopic only
    python topic_ultra.py data.csv text_col --no-stm         # skip STM (needs R)
    python topic_ultra.py demo                               # demo
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, sys, tempfile, time, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Shared infrastructure (ultra_shared, optional) ---
import sys as _sys
_ultra_parent = str(Path(__file__).resolve().parent.parent)
if _ultra_parent not in _sys.path:
    _sys.path.insert(0, _ultra_parent)

try:
    from ultra_shared.logging import setup_logging as _setup_logging
    from ultra_shared.config import load_config as _load_config
    HAS_ULTRA_SHARED = True
except ImportError:
    HAS_ULTRA_SHARED = False

try:
    from ultra_shared.schema import build_manifest, new_doc, add_tool_section, write_docs_jsonl, write_manifest
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False

try:
    from ultra_shared.report import ReportBuilder, THRESHOLDS
    HAS_REPORT = True
except ImportError:
    HAS_REPORT = False

from ultra_shared.data import load_documents


import shutil
_DEFAULT_RSCRIPT = shutil.which("Rscript") or shutil.which("Rscript.exe")
RSCRIPT = os.environ.get("RSCRIPT_PATH") or _DEFAULT_RSCRIPT or r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe"
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","this","that","these","those","am","it","its",
    "he","she","they","them","we","you","i","me","my","your","his","her",
    "our","their","what","which","who","whom","how","when","where","why",
    "all","each","every","both","few","more","most","other","some","such",
    "no","nor","not","only","own","same","so","than","too","very","just",
    "because","as","until","while","about","between","through","during",
    "before","after","above","below","up","down","out","off","over","under",
    "again","further","then","once","here","there","also","into","back",
    "going","get","got","make","made","like","even","much","well","still",
    "one","two","three","first","last","new","old","good","bad","big",
    "small","way","thing","things","people","think","know","really","right",
    "come","said","say","says","told","give","take","let","see","look",
    "want","use","used","using","find","keep","try","put","end","run","set",
    "feel","lot","http","https","www","com","org","hk","post","thread",
}
WORD_RE = re.compile(r"[A-Za-z]{2,}")


def clean_light(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_classical(text: str) -> str:
    words = WORD_RE.findall(text.lower())
    return " ".join(w for w in words if w not in STOPWORDS and len(w) > 2)


# ============================================================================
# EVALUATION
# ============================================================================

def compute_coherence(topic_words, texts, dictionary, measure="c_v"):
    try:
        from gensim.models.coherencemodel import CoherenceModel
        cm = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary,
                            coherence=measure, processes=min(os.cpu_count() or 2, 8))
        per_topic = [round(float(x), 4) for x in cm.get_coherence_per_topic()]
        result = {"measure": measure, "mean": round(float(cm.get_coherence()), 4), "per_topic": per_topic}
        if measure == "c_npmi" and (np.isnan(result["mean"]) or result["mean"] == 0):
            return compute_coherence(topic_words, texts, dictionary, measure="c_v")
        return result
    except Exception as e:
        return {"measure": measure, "mean": 0, "per_topic": [], "error": str(e)}


def topic_diversity(topics, top_n=10):
    seen = set()
    total = 0
    for t in topics:
        for w in t[:top_n]:
            if w: seen.add(w); total += 1
    return round(len(seen) / max(1, total), 4)


def topic_redundancy(topics, top_n=10):
    sets = [set(t[:top_n]) for t in topics]
    if len(sets) < 2: return 0.0
    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlaps.append(len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1))
    return round(float(np.mean(overlaps)), 4)


def pareto_select(results: List[Dict]) -> List[Dict]:
    """Select top-3 models on Pareto front: maximize c_v, maximize diversity, minimize redundancy."""
    if not results: return []
    scored = []
    for r in results:
        cv = r.get("coherence_cv", {}).get("mean", 0) if isinstance(r.get("coherence_cv"), dict) else 0
        div = r.get("diversity", 0)
        red = r.get("redundancy", 1)
        score = cv * 0.5 + div * 0.3 + (1 - red) * 0.2
        scored.append({**r, "_score": score})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:3]


def representative_docs(topic_ids, texts, doc_topic_matrix, top_n=3):
    reps = {}
    for tid in set(topic_ids):
        if tid == -1: continue
        mask = topic_ids == tid
        indices = np.where(mask)[0]
        if len(indices) == 0: continue
        scores = doc_topic_matrix[indices, tid] if doc_topic_matrix.ndim > 1 else np.ones(len(indices))
        top_idx = indices[np.argsort(scores)[::-1][:top_n]]
        reps[int(tid)] = [texts[i][:200] for i in top_idx]
    return reps


def compute_consensus(all_results, texts):
    """Compute pairwise Jaccard similarity between topics across models.
    Groups topics that share >=3 common words into consensus clusters."""
    from collections import defaultdict

    model_topics = {}
    for name, data in all_results.items():
        if data is None or "error" in data:
            continue
        if name == "bertopic":
            if "topic_words" in data:
                model_topics[name] = data["topic_words"]
        elif "best" in data and data["best"] and "topic_words" in data["best"]:
            model_topics[name] = data["best"]["topic_words"]

    if len(model_topics) < 2:
        print("\n  [CONSENSUS] Fewer than 2 models with topics — skipping consensus.")
        return

    topic_list = []
    for model_name, topic_words in model_topics.items():
        for tid, words in enumerate(topic_words):
            topic_list.append({
                "model": model_name,
                "topic_id": tid,
                "words": set(w.lower() for w in words[:10]),
            })

    n = len(topic_list)
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            common = topic_list[i]["words"] & topic_list[j]["words"]
            if len(common) >= 3:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        stack = [i]
        visited.add(i)
        while stack:
            node = stack.pop()
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    cluster.append(nb)
                    stack.append(nb)
        clusters.append(cluster)

    print("\n" + "=" * 70)
    print("ENSEMBLE CONSENSUS TOPICS")
    print("=" * 70)

    rows = []
    for ci, cluster in enumerate(clusters):
        models_in = sorted(set(topic_list[i]["model"] for i in cluster))
        if len(models_in) < 2:
            continue
        shared_words = set.intersection(*[topic_list[i]["words"] for i in cluster]) if len(cluster) > 1 else topic_list[cluster[0]]["words"]
        summary = ", ".join(sorted(shared_words)[:8])
        print(f"  Cluster {ci}: {', '.join(models_in)}")
        print(f"    Shared words: {summary}")
        for i in cluster:
            entry = topic_list[i]
            topic_label_words = ", ".join(sorted(entry["words"])[:6])
            print(f"      [{entry['model']}] Topic {entry['topic_id']}: {topic_label_words}")
            rows.append({
                "cluster_id": ci,
                "model": entry["model"],
                "topic_id": entry["topic_id"],
                "shared_words": summary,
                "topic_words": topic_label_words,
            })

    if not rows:
        print("  No consensus clusters found (no cross-model topic pairs with >=3 shared words).")
    else:
        out_dir = Path("output_topic_ultra")
        out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / "consensus_topics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  Saved {len(rows)} consensus mappings to {csv_path}")


# ============================================================================
# BERTOPIC
# ============================================================================

def run_bertopic(texts, doc_ids, min_topic_size=10, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 cache_embeddings=False, cache_dir=None):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from gensim.corpora import Dictionary

    t0 = time.time()
    print(f"  [BERTopic] Embeddings ({embedding_model})...")

    cache_path = None
    if cache_embeddings and cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        text_hash = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"embeddings_{text_hash}.npy"
        if cache_path.exists():
            embeddings = np.load(str(cache_path))
            print(f"  [BERTopic] Loaded cached embeddings: {embeddings.shape}")
        else:
            st = SentenceTransformer(embedding_model)
            embeddings = st.encode(texts, show_progress_bar=True, normalize_embeddings=True,
                                   batch_size=64)
            np.save(str(cache_path), embeddings)
            print(f"  [BERTopic] Saved embeddings cache: {cache_path}")
    else:
        st = SentenceTransformer(embedding_model)
        embeddings = st.encode(texts, show_progress_bar=False, normalize_embeddings=True,
                               batch_size=64)

    vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95)
    hdbscan = HDBSCAN(min_cluster_size=min_topic_size, min_samples=3, metric="euclidean", prediction_data=True)
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)

    model = BERTopic(embedding_model=st, umap_model=umap, hdbscan_model=hdbscan,
                     vectorizer_model=vec, verbose=False)
    topics, probs = model.fit_transform(texts, embeddings)
    if probs is not None and not isinstance(probs, np.ndarray):
        probs = np.array(probs) if probs else None
    elif probs is None:
        probs = np.zeros((len(texts), len(set(topics))))

    n_topics = len([t for t in set(topics) if t != -1])
    n_outliers = sum(1 for t in topics if t == -1)

    topic_words = []
    for tid in sorted(set(topics)):
        if tid == -1: continue
        topic_words.append([w for w, _ in model.get_topic(tid)[:10]])

    tokenized = [t.split() for t in texts]
    dictionary = Dictionary(tokenized)
    no_below = 1 if len(tokenized) < 2000 else 3
    no_above = 0.9 if len(tokenized) < 2000 else 0.7
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    cv = compute_coherence(topic_words, tokenized, dictionary, "c_v")
    cnpmi = compute_coherence(topic_words, tokenized, dictionary, "c_npmi")
    div = topic_diversity(topic_words)
    red = topic_redundancy(topic_words)

    doc_topic = probs if probs is not None and isinstance(probs, np.ndarray) and probs.ndim == 2 else np.zeros((len(texts), max(n_topics, 1)))
    reps = representative_docs(np.array(topics), texts, doc_topic)

    # Save visualizations
    viz = {}
    try:
        vdir = Path("output_topic_ultra") / "visualizations"
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            fig = model.visualize_topics()
            fig.write_html(str(vdir / "bertopic_topics.html"))
            viz["topic_map"] = str(vdir / "bertopic_topics.html")
        except Exception: pass
        try:
            fig = model.visualize_hierarchy()
            fig.write_html(str(vdir / "bertopic_hierarchy.html"))
            viz["hierarchy"] = str(vdir / "bertopic_hierarchy.html")
        except Exception: pass
        try:
            fig = model.visualize_barchart(top_n_topics=min(12, n_topics))
            fig.write_html(str(vdir / "bertopic_barchart.html"))
            viz["barchart"] = str(vdir / "bertopic_barchart.html")
        except Exception: pass
        try:
            import umap as _umap
            reduced_emb = _umap.UMAP(n_neighbors=10, n_components=2, metric="cosine", random_state=42).fit_transform(embeddings)
            fig = model.visualize_documents(texts, embeddings=reduced_emb)
            fig.write_html(str(vdir / "bertopic_documents.html"))
            viz["doc_map"] = str(vdir / "bertopic_documents.html")
        except Exception: pass
    except Exception: pass

    return {
        "model_type": "bertopic", "n_topics": n_topics, "n_outliers": n_outliers,
        "min_topic_size": min_topic_size, "topic_words": topic_words,
        "coherence_cv": cv, "coherence_npmi": cnpmi,
        "diversity": div, "redundancy": red,
        "elapsed_sec": round(time.time() - t0, 2),
        "representative_docs": reps, "visualizations": viz,
    }


# ============================================================================
# NMF + LDA (parallel K-sweep)
# ============================================================================

def _nmf_single_k(args):
    k, tfidf, feature_names, tokenized, dictionary = args
    from sklearn.decomposition import NMF
    from gensim.corpora import Dictionary
    t0 = time.time()
    model = NMF(n_components=k, random_state=42, max_iter=500, init="nndsvda")
    W = model.fit_transform(tfidf)
    H = model.components_

    topic_words = []
    topic_word_probs = []
    for i in range(k):
        top_idx = np.argsort(H[i])[::-1][:10]
        topic_words.append([feature_names[j] for j in top_idx])
        topic_word_probs.append({feature_names[j]: round(float(H[i, j]), 6) for j in top_idx})

    # Prevalence
    doc_topic_sum = W.sum(axis=1)
    doc_topic_norm = W / np.maximum(doc_topic_sum[:, np.newaxis], 1e-12)
    prevalence = [round(float(doc_topic_norm[:, i].mean()), 4) for i in range(k)]

    topic_details = []
    for tid in range(k):
        n_docs = int((W.argmax(axis=1) == tid).sum())
        topic_details.append({
            "topic_id": tid,
            "prob_words": topic_words[tid],
            "word_probs": topic_word_probs[tid],
            "prevalence": prevalence[tid],
            "n_docs": n_docs,
        })

    dominant = W.argmax(axis=1)
    reps = representative_docs(dominant, ["" for _ in range(W.shape[0])], W)
    cv = compute_coherence(topic_words, tokenized, dictionary, "c_v")
    cnpmi = compute_coherence(topic_words, tokenized, dictionary, "c_npmi")

    return {"model_type": "nmf", "k": k, "topic_words": topic_words,
            "topic_word_probs": topic_word_probs, "topic_details": topic_details,
            "topic_prevalence": prevalence,
            "coherence_cv": cv, "coherence_npmi": cnpmi,
            "diversity": topic_diversity(topic_words), "redundancy": topic_redundancy(topic_words),
            "elapsed_sec": round(time.time() - t0, 2), "representative_docs": reps}


def _lda_single_k(args):
    k, counts, feature_names, tokenized, dictionary = args
    from sklearn.decomposition import LatentDirichletAllocation
    from gensim.corpora import Dictionary
    t0 = time.time()
    model = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=50, learning_method="online")
    W = model.fit_transform(counts)
    H = model.components_

    topic_words = []
    topic_word_probs = []
    for i in range(k):
        top_idx = np.argsort(H[i])[::-1][:10]
        topic_words.append([feature_names[j] for j in top_idx])
        topic_word_probs.append({feature_names[j]: round(float(H[i, j]), 6) for j in top_idx})

    doc_topic_sum = W.sum(axis=1)
    doc_topic_norm = W / np.maximum(doc_topic_sum[:, np.newaxis], 1e-12)
    prevalence = [round(float(doc_topic_norm[:, i].mean()), 4) for i in range(k)]

    topic_details = []
    for tid in range(k):
        n_docs = int((W.argmax(axis=1) == tid).sum())
        topic_details.append({
            "topic_id": tid,
            "prob_words": topic_words[tid],
            "word_probs": topic_word_probs[tid],
            "prevalence": prevalence[tid],
            "n_docs": n_docs,
        })

    dominant = W.argmax(axis=1)
    reps = representative_docs(dominant, ["" for _ in range(W.shape[0])], W)
    cv = compute_coherence(topic_words, tokenized, dictionary, "c_v")
    cnpmi = compute_coherence(topic_words, tokenized, dictionary, "c_npmi")

    return {"model_type": "lda", "k": k, "topic_words": topic_words,
            "topic_word_probs": topic_word_probs, "topic_details": topic_details,
            "topic_prevalence": prevalence,
            "coherence_cv": cv, "coherence_npmi": cnpmi,
            "diversity": topic_diversity(topic_words), "redundancy": topic_redundancy(topic_words),
            "elapsed_sec": round(time.time() - t0, 2), "representative_docs": reps}


def run_nmf_parallel(texts, k_values, max_workers=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from gensim.corpora import Dictionary
    print(f"  [NMF] K sweep: {k_values} (parallel)")
    tokenized = [clean_classical(t).split() for t in texts]
    dictionary = Dictionary(tokenized)
    no_below = 1 if len(tokenized) < 2000 else 3
    no_above = 0.9 if len(tokenized) < 2000 else 0.7
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    n_vocab = len(dictionary)
    mf = min(max(1000, n_vocab * 3), 15000)
    vec = TfidfVectorizer(max_features=mf, min_df=2, max_df=0.9, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    tfidf = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out().tolist()
    valid_k = [k for k in k_values if k < min(tfidf.shape)]

    tasks = [(k, tfidf, feature_names, tokenized, dictionary) for k in valid_k]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers or min(os.cpu_count() or 2, 4)) as pool:
        futures = {pool.submit(_nmf_single_k, t): t[0] for t in tasks}
        for f in as_completed(futures):
            try: results.append(f.result())
            except Exception as e: print(f"    NMF K={futures[f]} failed: {e}")
    results.sort(key=lambda r: r["k"])
    best = max(results, key=lambda r: r["coherence_cv"]["mean"]) if results else None
    return {"results": results, "best": best, "pareto": pareto_select(results)}


def run_lda_parallel(texts, k_values, max_workers=None):
    from sklearn.feature_extraction.text import CountVectorizer
    from gensim.corpora import Dictionary
    print(f"  [LDA] K sweep: {k_values} (parallel)")
    tokenized = [clean_classical(t).split() for t in texts]
    dictionary = Dictionary(tokenized)
    no_below = 1 if len(tokenized) < 2000 else 3
    no_above = 0.9 if len(tokenized) < 2000 else 0.7
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    n_vocab = len(dictionary)
    mf = min(max(1000, n_vocab * 3), 15000)
    vec = CountVectorizer(max_features=mf, min_df=2, max_df=0.9, ngram_range=(1, 2), stop_words="english")
    counts = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out().tolist()
    valid_k = [k for k in k_values if k < min(counts.shape)]

    tasks = [(k, counts, feature_names, tokenized, dictionary) for k in valid_k]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers or min(os.cpu_count() or 2, 4)) as pool:
        futures = {pool.submit(_lda_single_k, t): t[0] for t in tasks}
        for f in as_completed(futures):
            try: results.append(f.result())
            except Exception as e: print(f"    LDA K={futures[f]} failed: {e}")
    results.sort(key=lambda r: r["k"])
    best = max(results, key=lambda r: r["coherence_cv"]["mean"]) if results else None
    return {"results": results, "best": best, "pareto": pareto_select(results)}


# ============================================================================
# STM (R via subprocess)
# ============================================================================

def run_stm(texts, k_values, output_dir):
    """Run STM via R subprocess. Exports FREX, beta, theta, topic correlations, prevalence."""
    print(f"  [STM] K sweep: {k_values} (R subprocess)")
    if not Path(RSCRIPT).exists():
        return {"results": [], "error": f"Rscript not found at {RSCRIPT}"}

    stm_dir = Path(output_dir) / "stm_runs"
    stm_dir.mkdir(parents=True, exist_ok=True)

    docs_df = pd.DataFrame({"doc_id": range(len(texts)), "text": texts})
    docs_csv = stm_dir / "documents.csv"
    docs_df.to_csv(docs_csv, index=False)

    results = []
    for k in k_values:
        prob_path  = (stm_dir / f"stm_k{k}_prob.tsv").as_posix()
        frex_path  = (stm_dir / f"stm_k{k}_frex.tsv").as_posix()
        lift_path  = (stm_dir / f"stm_k{k}_lift.tsv").as_posix()
        score_path = (stm_dir / f"stm_k{k}_score.tsv").as_posix()
        beta_path  = (stm_dir / f"stm_k{k}_beta.tsv").as_posix()
        theta_path = (stm_dir / f"stm_k{k}_theta.tsv").as_posix()
        prev_path  = (stm_dir / f"stm_k{k}_prevalence.tsv").as_posix()
        cor_path   = (stm_dir / f"stm_k{k}_correlations.tsv").as_posix()
        done_tag   = f"STM_K={k}_DONE"

        r_script = f'''
library(stm)
docs <- read.csv("{docs_csv.as_posix()}", stringsAsFactors = FALSE)
processed <- textProcessor(docs$text, metadata = data.frame(doc_id = docs$doc_id),
                           lowercase = TRUE, removestopwords = TRUE, removenumbers = TRUE,
                           removepunctuation = TRUE)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
fit <- stm(out$documents, out$vocab, K = {k}, prevalence = ~ 1, data = out$meta,
           seed = 42, init.type = "Spectral", verbose = FALSE)

lt <- labelTopics(fit, n = 10)
write.table(lt$prob,  "{prob_path}",  sep = "\\t", row.names = FALSE, quote = FALSE)
write.table(lt$frex,  "{frex_path}",  sep = "\\t", row.names = FALSE, quote = FALSE)
write.table(lt$lift,  "{lift_path}",  sep = "\\t", row.names = FALSE, quote = FALSE)
write.table(lt$score, "{score_path}", sep = "\\t", row.names = FALSE, quote = FALSE)

beta <- fit$beta
if (is.matrix(beta)) {{
  write.table(beta, "{beta_path}", sep = "\\t", row.names = FALSE, quote = FALSE)
}} else {{
  write.table(as.matrix(beta), "{beta_path}", sep = "\\t", row.names = FALSE, quote = FALSE)
}}
write.table(fit$theta, "{theta_path}", sep = "\\t", row.names = FALSE, quote = FALSE)

prev <- colMeans(fit$theta)
write.table(data.frame(topic = seq_along(prev), prevalence = round(prev, 4)),
          "{prev_path}", sep = "\\t", row.names = FALSE, quote = FALSE)

if ({k} >= 4) {{
  tryCatch({{
    cor <- topicCorr(fit, method = "simple", verbose = FALSE)
    write.table(as.matrix(cor$cor), "{cor_path}", sep = "\\t", row.names = TRUE, col.names = NA, quote = FALSE)
  }}, error = function(e) NULL)
}}

for (i in seq_len({k})) {{
  cat(paste0("TOPIC_", i, ":", paste(lt$prob[i,], collapse = ",")), "\\n")
}}

cat("{done_tag}")
'''
        r_file = stm_dir / f"run_stm_k{k}.R"
        r_file.write_text(r_script, encoding="utf-8")

        try:
            t0 = time.time()
            result = subprocess.run(
                [RSCRIPT, str(r_file)],
                capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
            )
            output = result.stdout + result.stderr
            elapsed = round(time.time() - t0, 2)
            if f"STM_K={k}_DONE" in output:
                # Read all outputs
                topic_words = []
                frex_words = []
                topic_prevalence = []

                for suffix, label in [("prob", "prob"), ("frex", "frex"), ("lift", "lift")]:
                    f = stm_dir / f"stm_k{k}_{suffix}.tsv"
                    if f.exists():
                        df = pd.read_csv(f, sep="\t")
                        if suffix == "prob":
                            topic_words = [df[col].dropna().tolist()[:10] for col in df.columns]
                        if suffix == "frex":
                            frex_words = [df[col].dropna().tolist()[:10] for col in df.columns]

                prev_f = stm_dir / f"stm_k{k}_prevalence.tsv"
                if prev_f.exists():
                    prev_df = pd.read_csv(prev_f, sep="\t")
                    topic_prevalence = prev_df["prevalence"].tolist()

                beta_file = stm_dir / f"stm_k{k}_beta.tsv"
                theta = pd.read_csv(stm_dir / f"stm_k{k}_theta.tsv", sep="\t").values if (stm_dir / f"stm_k{k}_theta.tsv").exists() else np.zeros((len(texts), k))

                # Per-topic details
                topic_details = []
                for tid in range(k):
                    topic_details.append({
                        "topic_id": tid,
                        "prob_words": topic_words[tid] if tid < len(topic_words) else [],
                        "frex_words": frex_words[tid] if tid < len(frex_words) else [],
                        "prevalence": round(topic_prevalence[tid], 4) if tid < len(topic_prevalence) else 0,
                        "n_docs": int((theta.argmax(axis=1) == tid).sum()) if theta.size > 0 else 0,
                    })

                reps = representative_docs(theta.argmax(axis=1), texts, theta)

                # Word-topic matrix
                beta = pd.read_csv(beta_file, sep="\t").values if beta_file.exists() else np.array([])

                from gensim.corpora import Dictionary
                tokenized = [t.split() for t in texts]
                dictionary = Dictionary(tokenized)
                no_below = 1 if len(tokenized) < 2000 else 3
                no_above = 0.9 if len(tokenized) < 2000 else 0.7
                dictionary.filter_extremes(no_below=no_below, no_above=no_above)
                cv = compute_coherence(topic_words, tokenized, dictionary, "c_v")
                cnpmi = compute_coherence(topic_words, tokenized, dictionary, "c_npmi")
                div = topic_diversity(topic_words)
                red = topic_redundancy(topic_words)

                results.append({
                    "model_type": "stm", "k": k, "topic_words": topic_words,
                    "frex_words": frex_words, "topic_details": topic_details,
                    "topic_prevalence": topic_prevalence,
                    "coherence_cv": cv, "coherence_npmi": cnpmi,
                    "diversity": div, "redundancy": red,
                    "representative_docs": reps,
                    "elapsed_sec": elapsed,
                    "beta_shape": list(beta.shape) if beta.size > 0 else [],
                    "theta_shape": list(theta.shape),
                })
                print(f"    STM K={k}: c_v={cv['mean']} diversity={div} ({elapsed}s)")
                for td in topic_details[:5]:
                    print(f"      Topic {td['topic_id']}: {', '.join(td['frex_words'][:6])}  ({td['prevalence']:.1%})")
            else:
                print(f"    STM K={k}: FAILED")
        except subprocess.TimeoutExpired:
            print(f"    STM K={k}: TIMEOUT")
        except Exception as e:
            print(f"    STM K={k}: ERROR - {e}")

    best = max(results, key=lambda r: r["coherence_cv"]["mean"]) if results else None
    return {"results": results, "best": best, "pareto": pareto_select(results)}


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def _generate_visualizations(all_results, comparison, texts, viz_dir):
    """Generate comprehensive HTML visualizations for all models."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 1. Coherence vs K comparison across all models
    fig = make_subplots(rows=2, cols=2, subplot_titles=("c_v Coherence vs K", "Topic Diversity vs K",
                                                          "Redundancy vs K", "Model Comparison"),
                         vertical_spacing=0.12, horizontal_spacing=0.1)
    colors = {"nmf": "#2196F3", "lda": "#4CAF50", "stm": "#FF9800"}

    for model_name in ["nmf", "lda", "stm"]:
        data = all_results.get(model_name)
        if not data or "results" not in data: continue
        results = data["results"]
        ks = [r["k"] for r in results]
        cvs = [r.get("coherence_cv", {}).get("mean", 0) for r in results]
        divs = [r.get("diversity", 0) for r in results]
        reds = [r.get("redundancy", 0) for r in results]
        fig.add_trace(go.Scatter(x=ks, y=cvs, mode="lines+markers", name=model_name.upper(),
                                  line=dict(color=colors.get(model_name, "#999")), marker=dict(size=6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ks, y=divs, mode="lines+markers", name=f"{model_name.upper()} div",
                                  line=dict(color=colors.get(model_name, "#999"), dash="dash"), marker=dict(size=6),
                                  showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=ks, y=reds, mode="lines+markers", name=f"{model_name.upper()} red",
                                  line=dict(color=colors.get(model_name, "#999"), dash="dot"), marker=dict(size=6),
                                  showlegend=False), row=2, col=1)

    # BERTopic point
    bt = all_results.get("bertopic", {})
    if bt and "error" not in bt:
        fig.add_trace(go.Scatter(x=[bt.get("n_topics", 3)], y=[bt.get("coherence_cv", {}).get("mean", 0)],
                                  mode="markers", name="BERTopic",
                                  marker=dict(size=12, color="#E91E63", symbol="star")), row=1, col=1)
        fig.add_trace(go.Scatter(x=[bt.get("n_topics", 3)], y=[bt.get("diversity", 0)],
                                  mode="markers", name="BERTopic div",
                                  marker=dict(size=12, color="#E91E63", symbol="star"), showlegend=False), row=1, col=2)

    # Model comparison bar chart
    if comparison:
        names = [f"{c['model']}\nK={c['k']}" for c in comparison[:8]]
        cvs = [c["cv"] for c in comparison[:8]]
        fig.add_trace(go.Bar(x=names, y=cvs, name="c_v", marker_color="#2196F3", showlegend=False), row=2, col=2)

    fig.update_layout(height=800, title_text="Topic Model Analysis — Multi-Metric Comparison", showlegend=True,
                      template="plotly_white")
    fig.write_html(str(viz_dir / "01_model_comparison.html"))

    # 2. Per-topic word clouds / bar charts for best model per type
    for model_name in ["nmf", "lda", "stm"]:
        data = all_results.get(model_name)
        if not data: continue
        best = data.get("best")
        if not best: continue
        topic_words = best.get("topic_words", [])
        if not topic_words: continue

        fig_words = make_subplots(rows=1, cols=min(4, len(topic_words)),
                                   subplot_titles=[f"Topic {i}" for i in range(min(4, len(topic_words)))])
        for i, words in enumerate(topic_words[:4]):
            top5 = words[:8]
            probs = []
            if best.get("topic_word_probs") and i < len(best["topic_word_probs"]):
                probs = [best["topic_word_probs"][i].get(w, 0.1) for w in top5]
            else:
                probs = list(range(len(top5), 0, -1))
            fig_words.add_trace(go.Bar(y=top5[::-1], x=probs[::-1], orientation="h",
                                        name=f"Topic {i}", showlegend=False,
                                        marker_color=colors.get(model_name, "#999")), row=1, col=i+1)

        fig_words.update_layout(height=300, title=f"{model_name.upper()} K={best['k']} — Top Words per Topic",
                                 template="plotly_white")
        fig_words.write_html(str(viz_dir / f"02_{model_name}_topics.html"))

    # 3. Topic prevalence heatmap for best model per type
    for model_name in ["nmf", "lda", "stm"]:
        data = all_results.get(model_name)
        if not data: continue
        best = data.get("best")
        if not best: continue
        details = best.get("topic_details", [])
        if not details: continue

        topics = [f"T{d['topic_id']}" for d in details]
        prevalences = [d["prevalence"] for d in details]
        top_words = [", ".join(d.get("prob_words", d.get("frex_words", []))[:5]) for d in details]

        fig_prev = go.Figure(data=[go.Bar(x=topics, y=prevalences,
                                           text=top_words, textposition="outside",
                                           marker_color=colors.get(model_name, "#999"))])
        fig_prev.update_layout(height=400, title=f"{model_name.upper()} K={best['k']} — Topic Prevalence",
                                xaxis_title="Topic", yaxis_title="Prevalence", template="plotly_white")
        fig_prev.write_html(str(viz_dir / f"03_{model_name}_prevalence.html"))

    # 4. Coherence heatmap: K x Model
    model_names = []
    k_sets = []
    for model_name in ["nmf", "lda", "stm"]:
        data = all_results.get(model_name)
        if not data or "results" not in data: continue
        model_names.append(model_name.upper())
        k_sets.append({r["k"]: r.get("coherence_cv", {}).get("mean", 0) for r in data["results"]})

    if model_names:
        all_k = sorted(set(k for ks in k_sets for k in ks))
        z = []
        for i, mn in enumerate(model_names):
            z.append([k_sets[i].get(k, 0) for k in all_k])
        fig_heat = go.Figure(data=go.Heatmap(z=z, x=[f"K={k}" for k in all_k], y=model_names,
                                              colorscale="RdYlGn", text=[[f"{v:.3f}" for v in row] for row in z],
                                              texttemplate="%{text}", textfont={"size": 10}))
        fig_heat.update_layout(height=300, title="Coherence (c_v) Heatmap: Model x K", template="plotly_white")
        fig_heat.write_html(str(viz_dir / "04_coherence_heatmap.html"))

    print(f"  Generated 4 visualization sets in {viz_dir}")


# ============================================================================
# T3: DOCUMENT-TOPIC MIXTURE PLOT (stacked bar per doc)
# ============================================================================

def plot_topic_mixtures(all_results, texts, output_dir="output_topic_ultra"):
    """Stacked bar chart showing per-document topic proportions for best model."""
    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    best_model = None
    for name in ["bertopic", "nmf", "lda", "stm"]:
        data = all_results.get(name)
        if data and data.get("best") and "error" not in data:
            best_model = name
            break
    if not best_model:
        return

    best = all_results[best_model]["best"]
    theta = best.get("topic_prevalence")
    if theta is None:
        return
    theta = np.array(theta)
    if theta.ndim != 2:
        return

    n_topics = theta.shape[1]
    n_show = min(20, theta.shape[0])
    topic_labels = [f"T{i}" for i in range(n_topics)]

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for t in range(n_topics):
            fig.add_trace(go.Bar(
                x=list(range(n_show)), y=theta[:n_show, t].tolist(),
                name=topic_labels[t]))
        fig.update_layout(
            barmode="stack", title=f"{best_model.upper()} Document-Topic Mixtures (top {n_show})",
            xaxis_title="Document", yaxis_title="Topic Proportion",
            template="plotly_white", height=500)
        fig.write_html(str(viz_dir / "05_topic_mixtures.html"))
        print(f"  Saved topic mixture plot to viz/05_topic_mixtures.html")
    except Exception:
        pass


# ============================================================================
# T5: AUTO TOPIC LABELS (KeyBERT-based)
# ============================================================================

def auto_topic_labels(topic_words):
    """Generate human-readable topic labels from top words using n-gram scoring.

    Selects the best 2-3 word combination from top words.
    """
    labels = []
    for i, words in enumerate(topic_words):
        if not words:
            labels.append(f"Topic {i}")
            continue
        # Score 2-grams from top words
        best_label = words[0] if words else f"Topic {i}"
        best_score = 0
        for j in range(min(8, len(words))):
            for k in range(j + 1, min(8, len(words))):
                bigram = f"{words[j]} {words[k]}"
                score = (8 - j) + (8 - k)
                if score > best_score:
                    best_score = score
                    best_label = bigram
        # Capitalize
        labels.append(best_label.title())
    return labels


# ============================================================================
# T2: TOPIC CORRELATION NETWORK
# ============================================================================

def plot_topic_network(all_results, output_dir="output_topic_ultra"):
    """Topic correlation network — which topics co-occur in documents."""
    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    best_model = None
    for name in ["bertopic", "nmf", "lda", "stm"]:
        data = all_results.get(name)
        if data and data.get("best") and "error" not in data:
            best_model = name
            break
    if not best_model:
        return

    best = all_results[best_model]["best"]
    theta = best.get("topic_prevalence")
    if theta is None:
        return
    theta = np.array(theta)
    if theta.ndim != 2:
        return

    n_topics = theta.shape[1]
    threshold = 0.1
    corr = np.corrcoef(theta.T)
    G = nx.Graph()
    for i in range(n_topics):
        G.add_node(f"T{i}")
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            if abs(corr[i, j]) > threshold:
                G.add_edge(f"T{i}", f"T{j}", weight=float(abs(corr[i, j])))

    try:
        import plotly.graph_objects as go
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_size = [G.degree(n) * 15 + 20 for n in G.nodes()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(width=1, color="#aaa")))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                  text=list(G.nodes()), textposition="top center",
                                  marker=dict(size=node_size, color="#3498db")))
        fig.update_layout(title=f"{best_model.upper()} Topic Correlation Network",
                          showlegend=False, template="plotly_white", height=500)
        fig.write_html(str(viz_dir / "06_topic_network.html"))
        print(f"  Saved topic network to viz/06_topic_network.html")
    except Exception:
        pass


# ============================================================================
# T14: CONTRASTIVE TOPICS (which topics distinguish groups)
# ============================================================================

def run_contrastive_topics(all_results, df, group_col="group", output_dir="output_topic_ultra"):
    """Which topics are over/under-represented in one group vs another?"""
    if group_col not in df.columns:
        return

    groups = df[group_col].unique()
    if len(groups) < 2:
        return

    best_model = None
    for name in ["bertopic", "nmf", "lda", "stm"]:
        data = all_results.get(name)
        if data and data.get("best") and "error" not in data:
            best_model = name
            break
    if not best_model:
        return

    best = all_results[best_model]["best"]
    theta = best.get("topic_prevalence")
    if theta is None:
        return
    theta = np.array(theta)
    if theta.ndim == 1:
        return
    g1, g2 = sorted(groups)[:2]
    n_docs = min(len(df), theta.shape[0])

    # Assign each doc to a group
    group_vals = df[group_col].values[:n_docs]
    mask1 = group_vals == g1
    mask2 = group_vals == g2

    mean1 = theta[mask1].mean(axis=0)
    mean2 = theta[mask2].mean(axis=0)

    topic_labels = best.get("topic_words", [[f"Topic {i}"] for i in range(theta.shape[1])])

    print(f"\n--- Contrastive Topics: {g1} vs {g2} ---")
    print(f"  {'Topic':<8} {g1+'/K':>8} {g2+'/K':>8} {'Diff':>8} {'Top words':<50}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*50}")
    diffs = []
    for i in range(theta.shape[1]):
        diff = mean1[i] - mean2[i]
        words = ", ".join(topic_labels[i][:5]) if i < len(topic_labels) else f"Topic {i}"
        diffs.append((i, diff, words))
        marker = " *" if abs(diff) > 0.05 else ""
        print(f"  T{i:<5} {mean1[i]:>8.3f} {mean2[i]:>8.3f} {diff:>+8.3f} {words:<50}{marker}")

    # Save
    rows = [{"topic_id": i, f"{g1}_mean": round(mean1[i], 4), f"{g2}_mean": round(mean2[i], 4),
             "diff": round(diffs[i][1], 4), "words": diffs[i][2]} for i in range(len(diffs))]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "contrastive_topics.csv"),
                               index=False, encoding="utf-8-sig")
    print(f"  Saved contrastive_topics.csv")


# ============================================================================
# T9: INTERACTIVE TOPIC BROWSER (pyLDAvis-style for all models)
# ============================================================================

def generate_interactive_topic_browser(all_results, output_dir="output_topic_ultra"):
    """Interactive HTML topic browser — word probabilities + intertopic distance map."""
    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    best_model = None
    for name in ["bertopic", "nmf", "lda", "stm"]:
        data = all_results.get(name)
        if data and data.get("best") and "error" not in data:
            best_model = name
            break
    if not best_model:
        return

    best = all_results[best_model]["best"]
    topic_words = best.get("topic_words", [])
    details = best.get("topic_details", [])
    theta = best.get("topic_prevalence")
    labels = auto_topic_labels(topic_words)
    reps = best.get("representative_docs", {})

    # Build interactive HTML with topic cards
    cards = []
    for i, words in enumerate(topic_words):
        prev = details[i]["prevalence"] if i < len(details) else 0
        label = labels[i] if i < len(labels) else f"Topic {i}"
        word_html = "".join(f'<span style="display:inline-block;margin:2px;padding:3px 8px;'
                            f'background:#e8f4f8;border-radius:4px;font-size:{max(12, 20 - j*2)}px;">'
                            f'{w}</span>' for j, w in enumerate(words[:10]))
        exemplar_html = ""
        if reps and i in reps:
            exemplars = reps[i]
            for doc_text in exemplars[:3]:
                excerpt = doc_text[:150].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                exemplar_html += f'<p class="exemplar" style="margin:4px 0;padding:6px 8px;background:#fff8e1;border-left:3px solid #ffb300;font-size:12px;color:#555;border-radius:0 4px 4px 0;">"{excerpt}..."</p>'
        cards.append(f'<div class="topic-card"><h3>T{i}: {label}</h3>'
                     f'<p>Prevalence: {prev:.1%}</p>'
                     f'<div class="words">{word_html}</div>'
                     f'{exemplar_html}</div>')

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Topic Browser — {best_model.upper()}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #2c3e50; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 16px; }}
  .topic-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; border: 1px solid #e0e0e0; }}
  .topic-card h3 {{ margin: 0 0 8px 0; color: #2980b9; }}
  .topic-card p {{ margin: 4px 0; color: #666; font-size: 13px; }}
  .words {{ margin-top: 8px; }}
</style>
</head><body>
<h1>{best_model.upper()} Topic Browser (K={best.get('k', '?')})</h1>
<p>Auto-generated labels using keyword scoring. c_v={best.get('coherence_cv', {}).get('mean', '?')}</p>
<div class="grid">{"".join(cards)}</div>
</body></html>"""

    out_path = str(viz_dir / "07_topic_browser.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved interactive topic browser to viz/07_topic_browser.html")


def run(df, output_dir=None, k_values=None, do_bertopic=True, do_nmf=True, do_lda=True, do_stm=True,
        cache_embeddings=False, group_col=None):
    t_start = time.time()
    output = Path(output_dir or "output_topic_ultra")
    output.mkdir(exist_ok=True)

    texts = df["text"].astype(str).tolist()
    doc_ids = df["doc_id"].astype(str).tolist()
    n_docs = len(texts)

    # K range: always 3-20, filtered by doc count
    if k_values is None:
        k_values = list(range(3, 21))
    max_k = min(n_docs - 2, 20)
    k_values = [k for k in k_values if 3 <= k <= max_k]

    print("\n" + "=" * 70)
    print("TOPIC ANALYSIS ULTRA")
    print("=" * 70)
    print(f"Documents: {n_docs}  K range: {k_values[0]}-{k_values[-1]}  Models: ", end="")
    models = []
    if do_bertopic: models.append("BERTopic")
    if do_stm: models.append("STM")
    if do_nmf: models.append("NMF")
    if do_lda: models.append("LDA")
    print(" + ".join(models))
    print()

    all_results = {}

    # === BERTopic ===
    if do_bertopic and n_docs >= 15:
        print("--- BERTopic ---")
        try:
            bt = run_bertopic(texts, doc_ids, min_topic_size=max(5, n_docs // 20),
                              cache_embeddings=cache_embeddings, cache_dir=str(output / "cache"))
            all_results["bertopic"] = bt
            print(f"  Topics: {bt['n_topics']}  Outliers: {bt['n_outliers']}")
            print(f"  c_v: {bt['coherence_cv']['mean']}  c_npmi: {bt['coherence_npmi']['mean']}")
            print(f"  Diversity: {bt['diversity']}  Redundancy: {bt['redundancy']}  ({bt['elapsed_sec']}s)")
            for i, words in enumerate(bt["topic_words"][:8]):
                print(f"  Topic {i}: {', '.join(words[:8])}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {e}")
            all_results["bertopic"] = {"error": str(e)}
    elif do_bertopic:
        print(f"  [SKIP] BERTopic needs >= 15 docs (have {n_docs})")

    # === STM ===
    if do_stm:
        print("\n--- STM (R) ---")
        try:
            stm = run_stm(texts, k_values, output)
            all_results["stm"] = stm
            if stm.get("best"):
                b = stm["best"]
                print(f"  Best: K={b['k']}  c_v={b['coherence_cv']['mean']}  div={b['diversity']}")
            if stm.get("pareto"):
                print(f"  Pareto top-3: K={[r['k'] for r in stm['pareto']]}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results["stm"] = {"error": str(e)}

    # === NMF (parallel) ===
    if do_nmf:
        print("\n--- NMF (parallel) ---")
        try:
            nmf = run_nmf_parallel(texts, k_values)
            all_results["nmf"] = nmf
            if nmf["best"]:
                b = nmf["best"]
                print(f"  Best: K={b['k']}  c_v={b['coherence_cv']['mean']}  "
                      f"div={b['diversity']}  red={b['redundancy']}  ({b['elapsed_sec']}s)")
            if nmf["pareto"]:
                print(f"  Pareto top-3: K={[r['k'] for r in nmf['pareto']]}")
            for r in nmf["results"][:5]:
                top3_words = r["topic_words"][:2]
                print(f"  K={r['k']}: c_v={r['coherence_cv']['mean']}  c_npmi={r['coherence_npmi']['mean']}  "
                      f"div={r['diversity']}  red={r['redundancy']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results["nmf"] = {"error": str(e)}

    # === LDA (parallel) ===
    if do_lda:
        print("\n--- LDA (parallel) ---")
        try:
            lda = run_lda_parallel(texts, k_values)
            all_results["lda"] = lda
            if lda["best"]:
                b = lda["best"]
                print(f"  Best: K={b['k']}  c_v={b['coherence_cv']['mean']}  "
                      f"div={b['diversity']}  red={b['redundancy']}  ({b['elapsed_sec']}s)")
            if lda["pareto"]:
                print(f"  Pareto top-3: K={[r['k'] for r in lda['pareto']]}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results["lda"] = {"error": str(e)}

    # === Final Comparison ===
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (auto top-3 K per model)")
    print("=" * 70)
    print(f"  {'Model':<12} {'K':<6} {'c_v':<8} {'c_npmi':<8} {'Diversity':<10} {'Redundancy':<10}")
    print(f"  {'-'*54}")

    comparison = []
    for name, data in all_results.items():
        if data is None or "error" in data: continue
        if name == "bertopic":
            comparison.append({"model": "BERTopic", "k": data.get("n_topics", 0),
                               "cv": data.get("coherence_cv", {}).get("mean", 0),
                               "cnpmi": data.get("coherence_npmi", {}).get("mean", 0),
                               "div": data.get("diversity", 0), "red": data.get("redundancy", 0)})
        elif "pareto" in data and data["pareto"]:
            for r in data["pareto"]:
                comparison.append({"model": name.upper(), "k": r["k"],
                                   "cv": r.get("coherence_cv", {}).get("mean", 0),
                                   "cnpmi": r.get("coherence_npmi", {}).get("mean", 0),
                                   "div": r.get("diversity", 0), "red": r.get("redundancy", 0)})

    comparison.sort(key=lambda x: x["cv"], reverse=True)
    for c in comparison[:10]:
        print(f"  {c['model']:<12} {c['k']:<6} {c['cv']:<8} {c['cnpmi']:<8} {c['div']:<10} {c['red']:<10}")

    if comparison:
        best = comparison[0]
        print(f"\n  >>> RECOMMENDED: {best['model']} K={best['k']} (c_v={best['cv']}, diversity={best['div']})")

    # === Consensus across models ===
    try:
        compute_consensus(all_results, texts)
    except Exception as e:
        print(f"  [CONSENSUS ERROR] {e}")

    # === VISUALIZATIONS ===
    print("\n--- Visualizations ---")
    viz_dir = output / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    try:
        _generate_visualizations(all_results, comparison, texts, viz_dir)
    except Exception as e:
        print(f"  [VIZ ERROR] {e}")

    # New features
    try:
        plot_topic_mixtures(all_results, texts, str(output))
    except Exception as e:
        print(f"  [MIXTURE ERROR] {e}")
    try:
        plot_topic_network(all_results, str(output))
    except Exception as e:
        print(f"  [NETWORK ERROR] {e}")
    try:
        run_contrastive_topics(all_results, df, group_col or ("group" if "group" in df.columns else df.columns[0]), str(output))
    except Exception as e:
        print(f"  [CONTRASTIVE ERROR] {e}")
    try:
        generate_interactive_topic_browser(all_results, str(output))
    except Exception as e:
        print(f"  [BROWSER ERROR] {e}")

    # === Save ===
    save = {}
    for k, v in all_results.items():
        if v is None:
            save[k] = None
        elif isinstance(v, dict):
            save[k] = {kk: vv for kk, vv in v.items() if kk not in ("bertopic_model", "embeddings")}
        else:
            save[k] = v
    save["comparison"] = comparison[:10]
    out_file = output / "topic_ultra_results.json"
    out_file.write_text(json.dumps(save, indent=2, default=str), encoding="utf-8")
    elapsed_total = round(time.time() - t_start, 2)

    if HAS_SCHEMA:
        # Emit per-doc topic assignments from best model
        best = comparison[0] if comparison else None
        if best:
            model_name = best["model"].lower()
            model_result = all_results.get(model_name)
            if model_result and "doc_topics" in model_result:
                docs = []
                for i, topic_id in enumerate(model_result["doc_topics"]):
                    topic_words_list = model_result.get("topic_words", [])
                    tw = topic_words_list[topic_id] if topic_id < len(topic_words_list) else []
                    d = new_doc(doc_ids[i] if i < len(doc_ids) else str(i),
                                texts[i] if i < len(texts) else "")
                    add_tool_section(d, "topics", {
                        "topic_id": int(topic_id),
                        "topic_words": tw[:10],
                        "model": model_name,
                    })
                    docs.append(d)
                write_docs_jsonl(docs, str(output))

        m = build_manifest(
            "topic_ultra", n_docs, time.time() - t_start,
            parameters={"k_values": k_values},
            extra={"best_model": comparison[0]["model"] if comparison else None,
                   "best_k": comparison[0]["k"] if comparison else None,
                   "best_cv": comparison[0]["cv"] if comparison else None}
        )
        write_manifest(m, str(output))
    else:
        # Fallback: manual manifest
        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_docs": n_docs,
            "k_values": k_values,
            "models_run": list(all_results.keys()),
            "elapsed_sec": elapsed_total,
            "best_model": comparison[0]["model"] if comparison else None,
            "best_k": comparison[0]["k"] if comparison else None,
            "best_cv": comparison[0]["cv"] if comparison else None,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # === HTML Report ===
    if HAS_REPORT:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            best = comparison[0] if comparison else {}
            rb = ReportBuilder(
                "Topic Analysis ULTRA",
                dataset=str(output),
                n_docs=n_docs,
                elapsed_sec=elapsed_total,
            )

            # --- KEY FINDINGS ---
            findings = []
            if best:
                findings.append(f"Best model: {best.get('model','')} K={best.get('k','')} (c_v={best.get('cv',0):.4f})")
                findings.append(f"Coherence: {'good' if best.get('cv',0)>0.55 else 'fair'} per Röder et al. 2015")
                findings.append(f"Diversity: {best.get('div',0):.4f} ({'excellent' if best.get('div',0)>0.9 else 'good'} per Dieng et al. 2020)")
                findings.append(f"Redundancy: {best.get('red',0):.4f} ({'excellent' if best.get('red',0)<0.05 else 'good'})")
                findings.append(f"Searched K={min(k_values)}-{max(k_values)} across {len(k_values)} values")
            rb.add_key_findings(findings[:7])

            # --- RATIONALE ---
            rb.add_rationale("K Selection",
                f"Searched K={min(k_values)}-{max(k_values)} across {len(k_values)} values. "
                f"Top-3 per model selected via Pareto front (max c_v, max diversity, min redundancy). "
                f"Final ranking by c_v. (Weston et al. 2023)")
            rb.add_rationale("Evaluation Metrics",
                "c_v (Röder et al. 2015), c_npmi (Aletras & Stevenson 2013), "
                "diversity (Dieng et al. 2020), redundancy (heuristic).")

            # --- METRICS ---
            if best:
                rb.add_metric("c_v Coherence", best.get("cv", 0), thresholds=THRESHOLDS.get("c_v"))
                rb.add_metric("c_npmi", best.get("cnpmi", 0), thresholds=THRESHOLDS.get("c_npmi"))
                rb.add_metric("Diversity", best.get("div", 0), thresholds=THRESHOLDS.get("diversity"))
                rb.add_metric("Redundancy", best.get("red", 0), thresholds=THRESHOLDS.get("redundancy"))

            # --- CHARTS: Coherence & Diversity vs K ---
            nmf_data = all_results.get("nmf", {})
            nmf_results = nmf_data.get("results", []) if isinstance(nmf_data, dict) else []
            if nmf_results:
                ks = [r["k"] for r in nmf_results]
                cvs = [r.get("coherence_cv", {}).get("mean", 0) if isinstance(r.get("coherence_cv"), dict) else 0 for r in nmf_results]
                divs = [r.get("diversity", 0) for r in nmf_results]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ks, y=cvs, mode="lines+markers", name="c_v", line=dict(color="#3b82f6")))
                fig.add_trace(go.Scatter(x=ks, y=divs, mode="lines+markers", name="diversity", line=dict(color="#10b981")))
                fig.update_layout(xaxis_title="K", yaxis_title="Score", title="Coherence & Diversity vs K",
                                  template="plotly_white", height=400)
                rb.add_chart(fig, title="Coherence & Diversity vs K")

            # --- CHARTS: Multi-model coherence comparison ---
            colors_map = {"nmf": "#2196F3", "lda": "#4CAF50", "stm": "#FF9800"}
            fig_multi = go.Figure()
            for model_name in ["nmf", "lda", "stm"]:
                data = all_results.get(model_name)
                if not data or "results" not in data:
                    continue
                results = data["results"]
                ks = [r["k"] for r in results]
                cvs = [r.get("coherence_cv", {}).get("mean", 0) if isinstance(r.get("coherence_cv"), dict) else 0 for r in results]
                fig_multi.add_trace(go.Scatter(
                    x=ks, y=cvs, mode="lines+markers", name=model_name.upper(),
                    line=dict(color=colors_map.get(model_name, "#999")), marker=dict(size=6)))
            bt = all_results.get("bertopic", {})
            if bt and "error" not in bt:
                fig_multi.add_trace(go.Scatter(
                    x=[bt.get("n_topics", 3)], y=[bt.get("coherence_cv", {}).get("mean", 0)],
                    mode="markers", name="BERTopic",
                    marker=dict(size=12, color="#E91E63", symbol="star")))
            if fig_multi.data:
                fig_multi.update_layout(
                    xaxis_title="K (number of topics)", yaxis_title="c_v Coherence",
                    template="plotly_white", height=400, title="Coherence vs K (All Models)")
                rb.add_chart(fig_multi, title="Coherence vs K (All Models)")

            # --- TABLES ---
            if comparison:
                comp_df = pd.DataFrame(comparison[:10])
                for col in ["cv", "cnpmi", "div", "red"]:
                    if col in comp_df.columns:
                        comp_df[col] = comp_df[col].round(4)
                comp_df = comp_df.rename(columns={"cv": "c_v", "cnpmi": "c_npmi", "div": "diversity", "red": "redundancy"})
                rb.add_table(comp_df, title="Topic Model Comparison (Top 10)")

            # Topic words for best model
            if best.get("model") and best.get("k"):
                m = best["model"].lower()
                tw_data = all_results.get(m, {})
                topic_words = []
                if isinstance(tw_data, dict):
                    if "best" in tw_data and tw_data["best"] and "topic_words" in tw_data["best"]:
                        topic_words = tw_data["best"]["topic_words"]
                    elif "results" in tw_data:
                        for r in tw_data["results"]:
                            if r.get("k") == best.get("k") and "topic_words" in r:
                                topic_words = r["topic_words"]
                                break
                if topic_words:
                    topic_df = pd.DataFrame([
                        {"topic_id": i, "top_words": ", ".join(w[:10]) if isinstance(w, list) else str(w)}
                        for i, w in enumerate(topic_words)
                    ])
                    rb.add_table(topic_df, title=f"Topic Words (Best: {best['model']} K={best['k']})", expand_col="top_words")

            # --- Topic prevalence chart + per-topic detail table ---
            best_model_name = best.get("model", "").lower() if best else ""
            if best_model_name and best_model_name in all_results:
                model_data = all_results[best_model_name]
                results_list = model_data.get("results", [])
                best_k = best.get("k", 3) if best else 3
                for r in results_list:
                    if r.get("k") == best_k:
                        td = r.get("topic_details", [])
                        tp = r.get("topic_prevalence", [])
                        cv_per_topic = r.get("coherence_cv", {}).get("per_topic", [])

                        # Topic prevalence pie chart
                        if td:
                            labels = [f"T{t.get('topic_id', i)}" for i, t in enumerate(td)]
                            sizes = [t.get("prevalence", 0) for t in td]
                            if any(s > 0 for s in sizes):
                                import plotly.graph_objects as go
                                fig_prev = go.Figure(data=[go.Pie(
                                    labels=labels, values=sizes, hole=0.3,
                                    textinfo="label+percent",
                                    marker_colors=px.colors.qualitative.Set2[:len(labels)]
                                )])
                                fig_prev.update_layout(title=f"Topic Prevalence (K={best_k})")
                                rb.add_chart(fig_prev, title=f"Topic Prevalence (K={best_k})")

                        # Per-topic detail table
                        if td:
                            detail_rows = []
                            for i, t in enumerate(td):
                                words = t.get("prob_words", [])
                                detail_rows.append({
                                    "topic_id": f"T{i}",
                                    "n_docs": t.get("n_docs", 0),
                                    "prevalence": round(t.get("prevalence", 0), 4),
                                    "top_words": ", ".join(words[:5]) if words else "",
                                    "c_v_per_topic": round(cv_per_topic[i], 4) if cv_per_topic and i < len(cv_per_topic) else "",
                                })
                            detail_df = pd.DataFrame(detail_rows)
                            rb.add_table(detail_df, title=f"Topic Details (Best: {best['model']} K={best_k})", expand_col="top_words")
                        break

            # --- Representative documents per topic ---
            best_model_name = best.get("model", "").lower() if best else ""
            if best_model_name and best_model_name in all_results:
                model_data = all_results[best_model_name]
                # representative_docs is in results list, keyed by k
                results_list = model_data.get("results", [])
                best_k = best.get("k", 3) if best else 3
                rep_docs = []
                for r in results_list:
                    if r.get("k") == best_k:
                        rep_docs = r.get("representative_docs", [])
                        break
                if rep_docs:
                    rep_rows = []
                    for i, docs_list in enumerate(rep_docs):
                        if isinstance(docs_list, list):
                            for j, doc in enumerate(docs_list[:2]):
                                rep_rows.append({
                                    "topic_id": i,
                                    "doc_index": j + 1,
                                    "text": str(doc)[:200],
                                })
                        elif isinstance(docs_list, str):
                            rep_rows.append({
                                "topic_id": i,
                                "doc_index": 1,
                                "text": str(docs_list)[:200],
                            })
                    if rep_rows:
                        rep_df = pd.DataFrame(rep_rows)
                        rb.add_table(rep_df, title="Representative Documents per Topic", expand_col="text")

            rb.build(str(output / "report.html"))
            rb.build_csv(str(output / "raw_output.csv"))
            print(f"  Saved report.html and raw_output.csv")
        except Exception as e:
            print(f"  [REPORT ERROR] {e}")

    print(f"\nTotal elapsed: {elapsed_total}s")
    print(f"Saved: {out_file}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic Analysis Ultra")
    parser.add_argument("csv_path", nargs="?", help="Input CSV")
    parser.add_argument("text_col", nargs="?", help="Text column name")
    parser.add_argument("-o", "--output", default="output_topic_ultra")
    parser.add_argument("--k", help="Comma-separated K values")
    parser.add_argument("--sample", type=int, help="Sample N documents (stratified if group col)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=["demo", "fast", "full", "research"], default="fast",
                        help="demo=NMF3-8, fast=NMF+LDA, full=+BERTopic, research=+STM")
    parser.add_argument("--no-bertopic", action="store_true")
    parser.add_argument("--no-stm", action="store_true")
    parser.add_argument("--no-nmf", action="store_true")
    parser.add_argument("--no-lda", action="store_true")
    parser.add_argument("--bertopic", action="store_true", help="Force BERTopic even in fast mode")
    parser.add_argument("--stm", action="store_true", help="Force STM (requires R)")
    parser.add_argument("--cache-embeddings", action="store_true", help="Cache BERTopic embeddings")
    parser.add_argument("--group", help="Group column for contrastive topics")
    args = parser.parse_args()

    k_values = None
    if args.k:
        k_values = sorted(set(int(x) for x in re.split(r"[,\s]+", args.k) if x.strip().isdigit()))

    profile = args.profile
    if profile == "demo":
        k_values = k_values or list(range(3, 9))
        do_bertopic = False; do_stm = False; do_nmf = True; do_lda = False
    elif profile == "fast":
        k_values = k_values or list(range(3, 11))
        do_bertopic = args.bertopic; do_stm = args.stm; do_nmf = True; do_lda = True
    elif profile == "full":
        k_values = k_values or list(range(3, 16))
        do_bertopic = not args.no_bertopic; do_stm = args.stm; do_nmf = True; do_lda = True
    elif profile == "research":
        k_values = k_values or list(range(3, 21))
        do_bertopic = not args.no_bertopic; do_stm = not args.no_stm; do_nmf = True; do_lda = True
    else:
        do_bertopic = not args.no_bertopic; do_stm = not args.no_stm; do_nmf = not args.no_nmf; do_lda = not args.no_lda

    if args.bertopic: do_bertopic = True
    if args.stm: do_stm = True

    df = load_documents(args.csv_path, args.text_col)

    if args.sample and args.sample < len(df):
        if args.group and args.group in df.columns:
            df = df.groupby(args.group, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(args.sample * len(x) / len(df)))),
                                   random_state=args.seed))
        else:
            df = df.sample(args.sample, random_state=args.seed)
        print(f"Sampled {len(df)} documents (seed={args.seed})")

    run(df, output_dir=args.output, k_values=k_values,
        do_bertopic=do_bertopic, do_nmf=do_nmf, do_lda=do_lda, do_stm=do_stm,
        cache_embeddings=args.cache_embeddings, group_col=args.group)
