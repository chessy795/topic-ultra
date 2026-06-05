import sys, time
sys.path.insert(0, r"C:\Users\mwp5a\Desktop\Paper\corpus_topic_lab\topic_ultra")
from topic_ultra import run_bertopic
import pandas as pd

df = pd.read_csv(r"C:\Users\mwp5a\Desktop\Paper\corpus_topic_lab\research_data\lonely_sample_10k.csv")
texts = df["text"].astype(str).tolist()
doc_ids = list(range(len(texts)))

t0 = time.time()
print(f"Starting BERTopic on {len(texts)} docs...")
try:
    result = run_bertopic(texts, doc_ids, min_topic_size=20)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    n_topics = result["n_topics"]
    n_outliers = result["n_outliers"]
    cv = result["coherence_cv"]["mean"]
    div = result["diversity"]
    print(f"Topics: {n_topics}, Outliers: {n_outliers}")
    print(f"c_v: {cv}, diversity: {div}")
    for i, words in enumerate(result["topic_words"][:10]):
        print(f"  T{i}: {', '.join(words[:8])}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
