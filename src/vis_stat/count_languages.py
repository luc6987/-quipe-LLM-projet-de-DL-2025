"""
Language Statistics for Text Features


This script analyzes language distribution in text fields (text, extended_tweet.full_text, quoted_status.text)
"""

import json
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.paths import DATA_DIR, get_output_dir

# Try to import language detection library
try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
    # Set seed for reproducibility
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available. Install with: pip install langdetect")

# Language code to name mapping (common languages)
LANG_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'cs': 'Czech',
    'ro': 'Romanian',
    'hu': 'Hungarian',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'uk': 'Ukrainian',
    'el': 'Greek',
    'he': 'Hebrew',
    'und': 'Undefined',
    'unknown': 'Unknown'
}


def detect_language(text: Any, min_length: int = 10) -> Optional[str]:
    """
    Detect language of text.
    
    Args:
        text: Text string
        min_length: Minimum text length for detection
        
    Returns:
        Language code or None
    """
    if not text or pd.isna(text):
        return None
    
    text_str = str(text).strip()
    
    # Skip very short texts
    if len(text_str) < min_length:
        return None
    
    if not LANGDETECT_AVAILABLE:
        return 'unknown'
    
    try:
        lang = detect(text_str)
        return lang
    except (LangDetectException, Exception):
        return 'unknown'


def get_lang_from_record(record: Dict, field_path: str) -> Optional[str]:
    """
    Get language from record field, checking both lang field and text content.
    
    Args:
        record: JSON record
        field_path: Field path like 'text', 'extended_tweet.full_text'
        
    Returns:
        Language code or None
    """
    # Method 1: Check if there's a lang field for this text field
    if field_path == 'text':
        if 'lang' in record:
            return record['lang']
    elif field_path == 'extended_tweet.full_text':
        if 'extended_tweet' in record and isinstance(record['extended_tweet'], dict):
            if 'lang' in record['extended_tweet']:
                return record['extended_tweet'].get('lang')
    elif field_path == 'quoted_status.text':
        if 'quoted_status' in record and isinstance(record['quoted_status'], dict):
            if 'lang' in record['quoted_status']:
                return record['quoted_status'].get('lang')
    
    # Method 2: Detect from text content
    parts = field_path.split('.')
    value = record
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    
    if value and isinstance(value, str) and len(value) > 10:
        return detect_language(value)
    
    return None


def analyze_language_distribution(file_path: Path, max_records: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze language distribution in text fields.
    
    Args:
        file_path: Path to JSONL file
        max_records: Maximum number of records to process (None for all)
        
    Returns:
        Dictionary with language statistics
    """
    print(f"\nAnalyzing language distribution in: {file_path.name}")
    print("=" * 80)
    
    text_fields = ['text', 'extended_tweet.full_text', 'quoted_status.text']
    
    # Statistics for each field
    field_stats = {field: {'lang_from_field': Counter(), 'lang_detected': Counter(), 
                           'total': 0, 'with_text': 0, 'missing': 0} 
                   for field in text_fields}
    
    # Combined statistics (any text field)
    combined_langs = Counter()
    
    total_records = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_records and line_num > max_records:
                break
            
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} records...")
            
            try:
                record = json.loads(line.strip())
                total_records += 1
                
                record_langs = set()
                
                for field in text_fields:
                    field_stats[field]['total'] += 1
                    
                    # Get language from lang field or detect
                    lang_from_field = get_lang_from_record(record, field)
                    
                    # Check if text exists
                    parts = field.split('.')
                    value = record
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    
                    if value and isinstance(value, str) and len(str(value).strip()) > 0:
                        field_stats[field]['with_text'] += 1
                        
                        # Use lang field if available, otherwise detect
                        if lang_from_field:
                            lang = lang_from_field
                            field_stats[field]['lang_from_field'][lang] += 1
                        else:
                            lang = detect_language(value)
                            if lang:
                                field_stats[field]['lang_detected'][lang] += 1
                        
                        if lang:
                            record_langs.add(lang)
                            field_stats[field]['lang_detected'][lang] += 1
                    else:
                        field_stats[field]['missing'] += 1
                
                # Add to combined
                for lang in record_langs:
                    combined_langs[lang] += 1
                    
            except json.JSONDecodeError:
                continue
    
    print(f"  Completed: {total_records:,} total records")
    
    return {
        'total_records': total_records,
        'field_stats': field_stats,
        'combined_langs': combined_langs
    }


def format_lang_name(lang_code: str) -> str:
    """Format language code to readable name."""
    if lang_code in LANG_NAMES:
        return f"{LANG_NAMES[lang_code]} ({lang_code})"
    return lang_code


def print_statistics(results: Dict[str, Any], dataset_name: str):
    """Print formatted statistics."""
    print("\n" + "=" * 80)
    print(f"LANGUAGE DISTRIBUTION STATISTICS: {dataset_name.upper()}")
    print("=" * 80)
    
    total_records = results['total_records']
    field_stats = results['field_stats']
    combined_langs = results['combined_langs']
    
    # Print combined statistics
    print(f"\n📊 COMBINED LANGUAGE DISTRIBUTION (Any Text Field)")
    print("-" * 80)
    print(f"Total Records: {total_records:,}")
    print(f"Records with Text (any field): {sum(combined_langs.values()):,}")
    
    if combined_langs:
        print(f"\nTop 20 Languages:")
        print(f"{'Rank':<6} {'Language':<35} {'Count':<12} {'Percentage':<12}")
        print("-" * 80)
        for rank, (lang, count) in enumerate(combined_langs.most_common(20), 1):
            pct = (count / total_records * 100) if total_records > 0 else 0
            lang_name = format_lang_name(lang)
            print(f"{rank:<6} {lang_name:<35} {count:<12,} {pct:>10.2f}%")
    
    # Print field-specific statistics
    for field, stats in field_stats.items():
        print(f"\n📝 FIELD: {field}")
        print("-" * 80)
        print(f"Total Records: {stats['total']:,}")
        print(f"Records with Text: {stats['with_text']:,}")
        print(f"Records Missing: {stats['missing']:,}")
        
        if stats['lang_detected']:
            print(f"\nTop 15 Languages:")
            print(f"{'Rank':<6} {'Language':<35} {'Count':<12} {'Percentage':<12}")
            print("-" * 80)
            for rank, (lang, count) in enumerate(stats['lang_detected'].most_common(15), 1):
                pct = (count / stats['with_text'] * 100) if stats['with_text'] > 0 else 0
                lang_name = format_lang_name(lang)
                print(f"{rank:<6} {lang_name:<35} {count:<12,} {pct:>10.2f}%")


def save_statistics(results: Dict[str, Any], output_path: Path):
    """Save statistics to JSON file."""
    output_dict = {
        'total_records': results['total_records'],
        'combined_languages': dict(results['combined_langs']),
        'field_statistics': {}
    }
    
    for field, stats in results['field_stats'].items():
        output_dict['field_statistics'][field] = {
            'total': stats['total'],
            'with_text': stats['with_text'],
            'missing': stats['missing'],
            'languages': dict(stats['lang_detected'])
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Statistics saved to: {output_path}")


def main():
    """Main function."""
    print("=" * 80)
    print("Language Distribution Analysis for Text Features")
    print("=" * 80)
    
    if not LANGDETECT_AVAILABLE:
        print("\n⚠️  Warning: langdetect library not available.")
        print("   Install with: pip install langdetect")
        print("   Will use lang fields from data if available.\n")
    
    # Get data files
    train_file = DATA_DIR / 'train.jsonl'
    test_file = DATA_DIR / 'kaggle_test.jsonl'
    
    # Get output directory
    output_dir = get_output_dir() / 'statistics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Process training data
    if train_file.exists():
        # Process sample first (faster)
        print("\n🔍 Quick Sample Analysis (first 10,000 records)...")
        sample_results = analyze_language_distribution(train_file, max_records=10000)
        print_statistics(sample_results, "Train (Sample)")
        
        # Process full dataset
        print("\n" + "=" * 80)
        print("Processing full training dataset...")
        print("=" * 80)
        full_results = analyze_language_distribution(train_file, max_records=None)
        print_statistics(full_results, "Train (Full)")
        all_results['train'] = full_results
        save_statistics(full_results, output_dir / 'language_statistics_train.json')
    else:
        print(f"⚠️  Training file not found: {train_file}")
    
    # Process test data (sample only)
    if test_file.exists():
        print("\n" + "=" * 80)
        print("Processing test dataset (sample)...")
        print("=" * 80)
        test_results = analyze_language_distribution(test_file, max_records=10000)
        print_statistics(test_results, "Test (Sample)")
        all_results['test'] = test_results
        save_statistics(test_results, output_dir / 'language_statistics_test.json')
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if 'train' in all_results:
        train_langs = all_results['train']['combined_langs']
        if train_langs:
            top_lang = train_langs.most_common(1)[0]
            print(f"\n✅ Training Data:")
            print(f"   Total Records: {all_results['train']['total_records']:,}")
            print(f"   Records with Text: {sum(train_langs.values()):,}")
            print(f"   Top Language: {format_lang_name(top_lang[0])} ({top_lang[1]:,} records, {top_lang[1]/all_results['train']['total_records']*100:.1f}%)")
            print(f"   Unique Languages: {len(train_langs)}")
            
            # Check if English is dominant
            if 'en' in train_langs:
                en_pct = train_langs['en'] / all_results['train']['total_records'] * 100
                print(f"\n📌 English Percentage: {en_pct:.1f}%")
                if en_pct > 90:
                    print("   → Data is predominantly English. BERTweet is recommended.")
                elif en_pct > 70:
                    print("   → Data is mostly English. Consider BERTweet for English, multilingual model for others.")
                else:
                    print("   → Data is multilingual. Use multilingual model (XLM-RoBERTa or mDeBERTa).")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

