import json
import re

def generate_plan():
    try:
        with open('output/preprocessing/visualizations/stats/feature_statistics_report.json', 'r') as f:
            report = json.load(f)
    except FileNotFoundError:
        print("Error: Report file not found.")
        return

    plan = {}

    # Specific overrides or known handling
    # According to spec: in_reply_to_status_id (70% missing) should NOT be imputed with mean
    # Instead, it should be converted to binary flag (is_reply)
    known_actions = {
        'label': {'action': 'target', 'method': 'none', 'reason': 'Target variable'},
        'challenge_id': {'action': 'drop', 'method': 'none', 'reason': 'ID for submission only'},
        'created_at': {'action': 'transform', 'method': 'extract_date_components', 'reason': 'Timestamp'},
        'quoted_status.created_at': {'action': 'drop', 'method': 'none', 'reason': 'High missing, potentially redundant timestamp'},
        'user.created_at': {'action': 'transform', 'method': 'extract_account_age', 'reason': 'Calculate account age'},
        'quoted_status.user.created_at': {'action': 'drop', 'method': 'none', 'reason': 'High missing timestamp'},
        # Broadcast signal: in_reply_to_status_id - convert to binary flag, do NOT impute
        'in_reply_to_status_id': {'action': 'transform', 'method': 'extract_binary_flag', 'reason': 'Broadcast signal: NaN->0 (broadcast), Value->1 (reply). Do NOT impute mean.'},
        'in_reply_to_status_id_str': {'action': 'transform', 'method': 'extract_binary_flag', 'reason': 'Broadcast signal: NaN->0 (broadcast), Value->1 (reply). Do NOT impute mean.'},
        # in_reply_to_user_id will be target encoded (handled separately in feature engineering)
        'in_reply_to_user_id': {'action': 'transform', 'method': 'target_encoding', 'reason': 'Target encoding for replied user ID'},
        'in_reply_to_user_id_str': {'action': 'transform', 'method': 'target_encoding', 'reason': 'Target encoding for replied user ID'},
        # Source: clean HTML and categorize (Mobile/Web/Pro Tools/Other)
        'source': {'action': 'transform', 'method': 'clean_and_categorize_source', 'reason': 'Clean HTML tags, categorize as Mobile/Web/Pro Tools/Other'},
        # Text fields: extract metadata before dropping
        'text': {'action': 'transform', 'method': 'extract_text_metadata', 'reason': 'Extract metadata (length, mentions, hashtags, URLs) before dropping'},
        'extended_tweet.full_text': {'action': 'transform', 'method': 'extract_text_metadata', 'reason': 'Extract metadata before dropping'},
        # User description: extract metadata (length, email, official words)
        'user.description': {'action': 'transform', 'method': 'extract_description_metadata', 'reason': 'Extract desc_len, has_email, has_official_words'},
        # User location: extract metadata (has_location, country/city)
        'user.location': {'action': 'transform', 'method': 'extract_location_metadata', 'reason': 'Extract has_location, country/city features'},
        # Geo/coordinates: binary flag
        'geo': {'action': 'transform', 'method': 'extract_binary_flag', 'reason': 'Binary flag: has_geo'},
        'coordinates': {'action': 'transform', 'method': 'extract_binary_flag', 'reason': 'Binary flag: has_coordinates'}
    }

    for feature in report:
        name = feature['feature_name']
        f_type = feature['feature_type']
        missing_pct = feature['missing_percentage']
        top_values = feature.get('top_values', [])
        
        # Check manual overrides first
        if name in known_actions:
            plan[name] = known_actions[name]
            continue

        # 1. Zero Variance / High Missing Checks
        if missing_pct >= 95.0:
            plan[name] = {
                'action': 'drop',
                'method': 'none',
                'reason': f'High missing percentage ({missing_pct:.2f}%)'
            }
            continue
        
        # Check for single value dominance (Zero Variance)
        if top_values:
            top_val_pct = top_values[0]['percentage']
            if top_val_pct >= 99.99: # Strict zero variance
                 plan[name] = {
                    'action': 'drop',
                    'method': 'none',
                    'reason': f'Zero variance (Top value {top_values[0]["value"]} is {top_val_pct:.2f}%)'
                }
                 continue

        # 2. ID / URL / Text Checks (Heuristics)
        if name.endswith('_id') or name.endswith('_id_str') or name.endswith('id'):
             # Exception: maybe keep some IDs if we do specific ID-based features, but generally drop for XGB
             # We'll check if it's a known useful ID. For now, drop raw IDs.
             plan[name] = {
                'action': 'drop',
                'method': 'none',
                'reason': 'ID feature'
            }
             continue
             
        if 'url' in name and 'count' not in name: # Drop url strings, keep counts
             plan[name] = {
                'action': 'drop',
                'method': 'none',
                'reason': 'URL string feature'
            }
             continue

        # Text fields: handled by known_actions for metadata extraction
        # Only drop quoted_status text fields (not used in spec)
        if name in ['quoted_status.text', 'quoted_status.extended_tweet.full_text']:
             plan[name] = {
                'action': 'drop',
                'method': 'none',
                'reason': 'Raw text content (quoted status, not used)'
            }
             continue
             
        # Description and location: handled by known_actions for metadata extraction
        # Only drop other text/PII fields
        if any(x in name for x in ['screen_name', 'name', 'full_name']):
             plan[name] = {
                'action': 'drop',
                'method': 'none',
                'reason': 'Free text/PII feature'
            }
             continue

        # 3. Feature Type Handling
        if f_type == 'numeric':
            if missing_pct > 0:
                plan[name] = {
                    'action': 'keep',
                    'method': 'impute_median',
                    'reason': 'Numeric feature with missing values'
                }
            else:
                plan[name] = {
                    'action': 'keep',
                    'method': 'passthrough',
                    'reason': 'Numeric feature'
                }
        
        elif f_type == 'binary':
             plan[name] = {
                'action': 'keep',
                'method': 'bool_to_int',
                'reason': 'Binary feature'
            }
            
        elif f_type == 'categorical':
            # Heuristic for cardinality based on report might be hard without exact counts
            # We assume categorical features kept this far are worth encoding
            # If it has missing values, we treat NaN as a category or impute
            plan[name] = {
                'action': 'transform',
                'method': 'frequency_encoding', # Safer default for high cardinality
                'reason': 'Categorical feature (default to freq encoding)'
            }
            # Source is handled by known_actions (clean_and_categorize_source)
            # Other categorical features use frequency encoding by default
                
        else:
            plan[name] = {
                'action': 'drop',
                'method': 'none',
                'reason': f'Unknown type: {f_type}'
            }

    # Save
    with open('src/xgboost/feature_processing_plan.json', 'w') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    print("Plan generated at src/xgboost/feature_processing_plan.json")

if __name__ == '__main__':
    generate_plan()
