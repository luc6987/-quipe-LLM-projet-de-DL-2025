import json

# Load misclassified
with open('output/shap/errors/validation/misclassified_samples.json') as f:
    data = json.load(f)

indices = [7, 11, 75, 91]
targets = []
for idx in indices:
    # Find in list
    for s in data['misclassified_samples']:
        if s['sample_index'] == idx:
            f = s['features']
            # Get values, handling defaults if missing
            s_val = f.get('user.statuses_count')
            l_val = f.get('user.listed_count')
            if s_val is not None and l_val is not None:
                targets.append({
                    'id': idx,
                    'statuses': float(s_val),
                    'listed': float(l_val),
                    'pred': s['predicted_label'],
                    'true': s['true_label']
                })
            break

print(f"Searching for {len(targets)} targets...")
dataset_matches = {}

with open('data/train.jsonl') as f:
    for i, line in enumerate(f):
        # Optimization: stop if all found
        if len(dataset_matches) == len(targets): break
        
        try:
            raw = json.loads(line)
            u = raw.get('user', {})
            s_count = u.get('statuses_count')
            l_count = u.get('listed_count')
            
            if s_count is None or l_count is None: continue
            
            s_count = float(s_count)
            l_count = float(l_count)
            
            for t in targets:
                if t['id'] in dataset_matches: continue
                
                if s_count == t['statuses'] and l_count == t['listed']:
                    # Found a match
                    print(f"=== MATCH ID {t['id']} (Line {i}) ===")
                    print(f"True: {t['true']}, Pred: {t['pred']}")
                    print(f"Text: {raw.get('text')}")
                    print(f"Desc: {u.get('description')}")
                    print(f"User: {u.get('screen_name')}")
                    print("-" * 30)
                    dataset_matches[t['id']] = True
        except:
            pass
