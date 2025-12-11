import json
import logging

# Load misclassified
with open('output/shap/errors/validation/misclassified_samples.json') as f:
    data = json.load(f)

indices = [7, 11]
target_samples = {s['sample_index']: s for s in data['misclassified_samples'] if s['sample_index'] in indices}

print("=== Misclassified Features (from JSON) ===")
for idx in indices:
    if idx in target_samples:
        s = target_samples[idx]
        f = s['features']
        print(f"Index: {idx}")
        print(f"  True Label: {s['true_label']}")
        print(f"  user.listed_count: {f.get('user.listed_count')}")
        print(f"  user.statuses_count: {f.get('user.statuses_count')}")
        print(f"  text_len: {len(f) if f else 0}") # Just to see

# Load raw from train.jsonl
print("\n=== Raw Data Features (from train.jsonl) ===")
with open('data/train.jsonl') as f:
    for i, line in enumerate(f):
        if i in indices:
            raw = json.loads(line)
            user = raw.get('user', {})
            print(f"Index: {i}")
            print(f"  Raw Label: {raw.get('label')}")
            print(f"  user.listed_count: {user.get('listed_count')}")
            print(f"  user.statuses_count: {user.get('statuses_count')}")
