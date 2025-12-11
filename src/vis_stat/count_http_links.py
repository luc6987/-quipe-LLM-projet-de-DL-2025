"""
Count HTTP Links in All Datasets
统计所有数据集中的HTTP链接

This script extracts and counts all HTTP/HTTPS links from train.jsonl and kaggle_test.jsonl.
"""

import json
import re
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any
import pandas as pd
from html.parser import HTMLParser

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.paths import DATA_DIR, get_output_dir

# HTTP/HTTPS URL pattern
URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)

# HTML href pattern
HREF_PATTERN = re.compile(r'href=["\'](https?://[^"\']+)["\']', re.IGNORECASE)


class HrefExtractor(HTMLParser):
    """Extract href URLs from HTML source."""
    def __init__(self):
        super().__init__()
        self.urls = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href' and value and value.startswith(('http://', 'https://')):
                    self.urls.append(value)


def extract_urls_from_text(text: Any) -> Set[str]:
    """Extract HTTP/HTTPS URLs from text using regex."""
    if not text or pd.isna(text):
        return set()
    
    text_str = str(text)
    urls = set(URL_PATTERN.findall(text_str))
    return urls


def extract_urls_from_html(html: Any) -> Set[str]:
    """Extract URLs from HTML href attributes."""
    if not html or pd.isna(html):
        return set()
    
    html_str = str(html)
    # Use regex first
    urls = set(HREF_PATTERN.findall(html_str))
    
    # Also try HTMLParser
    parser = HrefExtractor()
    try:
        parser.feed(html_str)
        urls.update(parser.urls)
    except:
        pass
    
    return urls


def extract_urls_from_dict(data: Dict, path: str = "") -> Set[str]:
    """Recursively extract URLs from dictionary values."""
    urls = set()
    
    if not isinstance(data, dict):
        return urls
    
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key
        
        if value is None:
            continue
        elif isinstance(value, str):
            # Check if it's a URL field
            if key in ['url', 'expanded_url', 'display_url', 'expanded', 'display']:
                if value.startswith(('http://', 'https://')):
                    urls.add(value)
            # Check if it's HTML (source field)
            elif key == 'source':
                urls.update(extract_urls_from_html(value))
            # Otherwise, extract from text
            else:
                urls.update(extract_urls_from_text(value))
        elif isinstance(value, dict):
            urls.update(extract_urls_from_dict(value, current_path))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    urls.update(extract_urls_from_dict(item, current_path))
                elif isinstance(item, str):
                    if item.startswith(('http://', 'https://')):
                        urls.add(item)
                    else:
                        urls.update(extract_urls_from_text(item))
    
    return urls


def extract_urls_from_record(record: Dict) -> Dict[str, Any]:
    """
    Extract all URLs from a single record and track their sources.
    Returns a dictionary with URLs and their source fields.
    """
    all_urls = {}
    
    # 1. Extract from entities.urls
    if 'entities' in record and isinstance(record['entities'], dict):
        if 'urls' in record['entities'] and isinstance(record['entities']['urls'], list):
            for url_obj in record['entities']['urls']:
                if isinstance(url_obj, dict):
                    for url_key in ['url', 'expanded_url', 'display_url']:
                        if url_key in url_obj and url_obj[url_key]:
                            url = url_obj[url_key]
                            if url.startswith(('http://', 'https://')):
                                if url not in all_urls:
                                    all_urls[url] = []
                                all_urls[url].append(f"entities.urls.{url_key}")
    
    # 2. Extract from quoted_status.entities.urls
    if 'quoted_status' in record and isinstance(record['quoted_status'], dict):
        quoted = record['quoted_status']
        if 'entities' in quoted and isinstance(quoted['entities'], dict):
            if 'urls' in quoted['entities'] and isinstance(quoted['entities']['urls'], list):
                for url_obj in quoted['entities']['urls']:
                    if isinstance(url_obj, dict):
                        for url_key in ['url', 'expanded_url', 'display_url']:
                            if url_key in url_obj and url_obj[url_key]:
                                url = url_obj[url_key]
                                if url.startswith(('http://', 'https://')):
                                    if url not in all_urls:
                                        all_urls[url] = []
                                    all_urls[url].append(f"quoted_status.entities.urls.{url_key}")
    
    # 3. Extract from text fields
    for text_field in ['text', 'extended_tweet.full_text']:
        field_parts = text_field.split('.')
        value = record
        for part in field_parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        
        if value and isinstance(value, str):
            urls = extract_urls_from_text(value)
            for url in urls:
                if url not in all_urls:
                    all_urls[url] = []
                all_urls[url].append(text_field)
    
    # 4. Extract from user profile fields
    if 'user' in record and isinstance(record['user'], dict):
        user = record['user']
        user_url_fields = ['url', 'profile_image_url', 'profile_image_url_https',
                          'profile_background_image_url', 'profile_background_image_url_https',
                          'profile_banner_url', 'description']
        
        for field in user_url_fields:
            if field in user and user[field]:
                value = user[field]
                if isinstance(value, str):
                    if field == 'description':
                        # Extract from description text
                        urls = extract_urls_from_text(value)
                        for url in urls:
                            if url not in all_urls:
                                all_urls[url] = []
                            all_urls[url].append(f"user.{field}")
                    elif value.startswith(('http://', 'https://')):
                        if value not in all_urls:
                            all_urls[value] = []
                        all_urls[value].append(f"user.{field}")
    
    # 5. Extract from source field (HTML)
    if 'source' in record and record['source']:
        urls = extract_urls_from_html(record['source'])
        for url in urls:
            if url not in all_urls:
                all_urls[url] = []
            all_urls[url].append('source')
    
    # 6. Extract from quoted_status_permalink
    if 'quoted_status_permalink' in record and isinstance(record['quoted_status_permalink'], dict):
        permalink = record['quoted_status_permalink']
        for key in ['url', 'expanded', 'display']:
            if key in permalink and permalink[key]:
                url = permalink[key]
                if isinstance(url, str) and url.startswith(('http://', 'https://')):
                    if url not in all_urls:
                        all_urls[url] = []
                    all_urls[url].append(f"quoted_status_permalink.{key}")
    
    # 7. General recursive extraction (catch any remaining URLs)
    general_urls = extract_urls_from_dict(record)
    for url in general_urls:
        if url not in all_urls:
            all_urls[url] = []
        all_urls[url].append('other')
    
    return all_urls


def process_jsonl_file(file_path: Path) -> Dict[str, Any]:
    """Process a JSONL file and extract all URLs."""
    print(f"\nProcessing {file_path.name}...")
    
    all_urls = defaultdict(list)  # url -> list of source fields
    total_records = 0
    records_with_urls = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  Processed {line_num:,} records...")
            
            try:
                record = json.loads(line.strip())
                total_records += 1
                
                urls_in_record = extract_urls_from_record(record)
                
                if urls_in_record:
                    records_with_urls += 1
                    for url, sources in urls_in_record.items():
                        all_urls[url].extend(sources)
            
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"  Completed: {total_records:,} total records, {records_with_urls:,} records with URLs")
    
    return {
        'total_records': total_records,
        'records_with_urls': records_with_urls,
        'urls': dict(all_urls)
    }


def categorize_url(url: str) -> str:
    """Categorize URL by type."""
    url_lower = url.lower()
    
    if 't.co/' in url_lower or 'twitter.com/i/web' in url_lower:
        return 'twitter_short_link'
    elif 'twitter.com' in url_lower:
        return 'twitter_full_link'
    elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'instagram.com' in url_lower:
        return 'instagram'
    elif 'facebook.com' in url_lower:
        return 'facebook'
    elif 'linkedin.com' in url_lower:
        return 'linkedin'
    elif 'twimg.com' in url_lower:
        return 'twitter_image'
    elif 'abs.twimg.com' in url_lower:
        return 'twitter_asset'
    elif url.startswith('http://'):
        return 'http'
    elif url.startswith('https://'):
        return 'https'
    else:
        return 'other'


def generate_statistics(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive statistics from extracted URLs."""
    stats = {
        'datasets': {},
        'overall': {
            'total_unique_urls': 0,
            'total_url_occurrences': 0,
            'url_categories': {},
            'top_urls': [],
            'field_distribution': {}
        }
    }
    
    all_unique_urls = set()
    all_url_occurrences = Counter()
    all_field_distribution = defaultdict(int)
    url_categories = Counter()
    
    for dataset_name, data in results.items():
        urls = data['urls']
        total_urls = sum(len(sources) for sources in urls.values())
        unique_urls = len(urls)
        
        dataset_stats = {
            'total_records': data['total_records'],
            'records_with_urls': data['records_with_urls'],
            'records_without_urls': data['total_records'] - data['records_with_urls'],
            'total_url_occurrences': total_urls,
            'unique_urls': unique_urls,
            'url_categories': {},
            'top_urls': [],
            'field_distribution': {}
        }
        
        # Count occurrences and categorize
        url_occurrences = Counter()
        field_dist = defaultdict(int)
        
        for url, sources in urls.items():
            count = len(sources)
            url_occurrences[url] = count
            all_url_occurrences[url] += count
            
            category = categorize_url(url)
            dataset_stats['url_categories'][category] = dataset_stats['url_categories'].get(category, 0) + count
            url_categories[category] += count
            
            for source in sources:
                field_dist[source] += 1
                all_field_distribution[source] += 1
            
            all_unique_urls.add(url)
        
        # Get top URLs
        top_urls = url_occurrences.most_common(20)
        dataset_stats['top_urls'] = [{'url': url, 'count': count} for url, count in top_urls]
        dataset_stats['field_distribution'] = dict(sorted(field_dist.items(), key=lambda x: x[1], reverse=True))
        
        stats['datasets'][dataset_name] = dataset_stats
    
    # Overall statistics
    stats['overall']['total_unique_urls'] = len(all_unique_urls)
    stats['overall']['total_url_occurrences'] = sum(all_url_occurrences.values())
    stats['overall']['url_categories'] = dict(url_categories)
    
    top_urls_overall = all_url_occurrences.most_common(30)
    stats['overall']['top_urls'] = [{'url': url, 'count': count} for url, count in top_urls_overall]
    
    stats['overall']['field_distribution'] = dict(sorted(all_field_distribution.items(), key=lambda x: x[1], reverse=True))
    
    return stats


def save_results(results: Dict[str, Dict[str, Any]], stats: Dict[str, Any], output_dir: Path):
    """Save results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics JSON
    stats_file = output_dir / 'http_links_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Statistics saved to: {stats_file}")
    
    # Save URL list CSV
    csv_rows = []
    for dataset_name, data in results.items():
        for url, sources in data['urls'].items():
            csv_rows.append({
                'dataset': dataset_name,
                'url': url,
                'occurrences': len(sources),
                'sources': '; '.join(set(sources)),
                'category': categorize_url(url)
            })
    
    df = pd.DataFrame(csv_rows)
    df = df.sort_values(['occurrences', 'url'], ascending=[False, True])
    
    csv_file = output_dir / 'http_links_list.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"✓ URL list saved to: {csv_file}")
    print(f"  Total rows: {len(df):,}")


def print_summary(stats: Dict[str, Any]):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print("HTTP Links Statistics Summary")
    print("="*80)
    
    for dataset_name, dataset_stats in stats['datasets'].items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Total records: {dataset_stats['total_records']:,}")
        print(f"  Records with URLs: {dataset_stats['records_with_urls']:,} ({dataset_stats['records_with_urls']/dataset_stats['total_records']*100:.1f}%)")
        print(f"  Records without URLs: {dataset_stats['records_without_urls']:,}")
        print(f"  Total URL occurrences: {dataset_stats['total_url_occurrences']:,}")
        print(f"  Unique URLs: {dataset_stats['unique_urls']:,}")
        
        print(f"\n  URL Categories:")
        for category, count in sorted(dataset_stats['url_categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {category}: {count:,}")
        
        print(f"\n  Top 10 URLs:")
        for i, item in enumerate(dataset_stats['top_urls'][:10], 1):
            print(f"    {i}. {item['url'][:70]}... ({item['count']:,} occurrences)")
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS:")
    print(f"{'='*80}")
    overall = stats['overall']
    print(f"  Total unique URLs across all datasets: {overall['total_unique_urls']:,}")
    print(f"  Total URL occurrences: {overall['total_url_occurrences']:,}")
    
    print(f"\n  Overall URL Categories:")
    for category, count in sorted(overall['url_categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {category}: {count:,}")
    
    print(f"\n  Top 10 URLs (across all datasets):")
    for i, item in enumerate(overall['top_urls'][:10], 1):
        print(f"    {i}. {item['url'][:70]}... ({item['count']:,} occurrences)")
    
    print(f"\n  Top 10 Source Fields:")
    for i, (field, count) in enumerate(list(overall['field_distribution'].items())[:10], 1):
        print(f"    {i}. {field}: {count:,}")


def main():
    """Main function."""
    print("="*80)
    print("HTTP Links Counter - Statistics for All Datasets")
    print("="*80)
    
    # Get data files
    train_file = DATA_DIR / 'train.jsonl'
    test_file = DATA_DIR / 'kaggle_test.jsonl'
    
    if not train_file.exists():
        print(f"Error: {train_file} not found!")
        return
    
    if not test_file.exists():
        print(f"Error: {test_file} not found!")
        return
    
    # Process files
    results = {}
    
    results['train'] = process_jsonl_file(train_file)
    results['test'] = process_jsonl_file(test_file)
    
    # Generate statistics
    print("\nGenerating statistics...")
    stats = generate_statistics(results)
    
    # Print summary
    print_summary(stats)
    
    # Save results
    output_dir = get_output_dir('statistics')
    save_results(results, stats, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()

