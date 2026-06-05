import json
with open(r"../output_lonely/topic/topic_ultra_results.json") as f:
    d = json.load(f)
for model in ["nmf", "lda", "bertopic", "stm"]:
    data = d.get(model)
    if not data or "error" in data:
        print(f"{model}: {'ERROR - ' + data.get('error','?')[:100] if data else 'not run'}")
        continue
    best = data.get("best", {})
    k = best.get("k", "?")
    cv = best.get("coherence_cv", {}).get("mean", "?")
    words = best.get("topic_words", [[]])[0][:6] if best.get("topic_words") else "?"
    print(f"{model.upper()}: K={k}, c_v={cv}, words={words}")
    # Show all topic words
    for i, tw in enumerate(best.get("topic_words", [])):
        print(f"  T{i}: {', '.join(tw[:8])}")
