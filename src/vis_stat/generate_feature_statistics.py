"""
Generate comprehensive feature statistics report in JSON format.
生成全面的特征统计报表（JSON格式）
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
from scipy import stats
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from datetime import datetime
import warnings

# Suppress pandas warnings about datetime parsing
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import data_processing as dp
from src.utils.paths import DATA_DIR, get_preprocessing_dir

logger = logging.getLogger(__name__)


def get_top_values(series: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Get top N most frequent values and their counts.
    获取出现频率最高的N个值及其计数。
    
    Returns original values (not encoded), including NaN if present.
    Optimized for performance - handles complex types efficiently.
    """
    # Quick check: if series is empty, return empty list
    if len(series) == 0:
        return []
    
    # Count all values including NaN
    # For complex types (lists, arrays), convert to string for counting
    try:
        # Fast path: try direct value_counts first
        value_counts = series.value_counts(dropna=False).head(top_n)
    except (TypeError, ValueError):
        # Slow path: for list/array types, sample first to avoid processing all rows
        # Only process a sample if the series is very large
        if len(series) > 50000:
            # Sample 10% for complex types to speed up
            sample_size = min(50000, len(series))
            sample_series = series.sample(n=sample_size, random_state=42)
            logger.debug(f"  Sampling {sample_size} rows for complex type column")
        else:
            sample_series = series
        
        # For complex types, use a faster approach
        # Check if we can use astype(str) directly (much faster)
        try:
            series_str = sample_series.astype(str)
        except:
            # Fallback: use apply but with limited processing
            # Only process first 10000 rows for very large series
            if len(sample_series) > 10000:
                sample_series = sample_series.head(10000)
            
            def safe_str(x):
                if pd.isna(x):
                    return x
                elif isinstance(x, (list, dict, np.ndarray)):
                    return f"<{type(x).__name__}>"  # Simplified representation
                else:
                    return str(x)
            
            series_str = sample_series.apply(safe_str)
        
        value_counts = series_str.value_counts(dropna=False).head(top_n)
    
    if len(value_counts) == 0:
        return []
    
    total_count = len(series)
    non_null_count = total_count - series.isna().sum()
    result = []
    
    for value, count in value_counts.items():
        # Preserve original value representation
        # Check for NaN: handle both scalar and array cases
        try:
            # For scalar values
            if isinstance(value, (list, np.ndarray, tuple)):
                # For array-like values, they are not NaN (they are actual values)
                is_na = False
            else:
                # For scalar values, check if NaN
                is_na = pd.isna(value)
                if isinstance(is_na, np.ndarray):
                    is_na = bool(is_na.all())
        except (TypeError, ValueError):
            is_na = False
        
        if is_na:
            value_str = "NaN"
            # Calculate percentage based on total (including NaN)
            percentage = float(count / total_count * 100)
        else:
            # Convert to string, preserving original format
            if isinstance(value, (list, np.ndarray)):
                # For array/list values, convert to string representation
                value_str = str(value)
            elif isinstance(value, (int, float)):
                # For numeric values, use original representation
                value_str = str(value)
            else:
                # For strings and other types, use string representation
                value_str = str(value)
            # Calculate percentage based on non-null values only
            percentage = float(count / non_null_count * 100) if non_null_count > 0 else 0.0
        
        result.append({
            "value": value_str,
            "frequency": int(count),
            "percentage": percentage
        })
    
    return result


def get_mode(series: pd.Series) -> Optional[Any]:
    """
    Get mode (most frequent value) if it exists.
    获取众数（如果存在）。
    """
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return None
    
    mode_result = stats.mode(non_null_series, keepdims=True)
    if len(mode_result.mode) > 0:
        mode_value = mode_result.mode[0]
        # Convert numpy types to Python native types
        if isinstance(mode_value, (np.integer, np.floating)):
            return mode_value.item()
        return mode_value
    return None


def detect_feature_type(series: pd.Series) -> str:
    """
    Detect feature type: numeric, category, binary, etc.
    检测特征类型：数值型、类别型、二元型等。
    """
    dtype = series.dtype
    
    # Check if numeric
    if pd.api.types.is_numeric_dtype(dtype):
        # Check if binary (only 0 and 1)
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return "binary"
        # Check if integer with few unique values (likely categorical)
        if pd.api.types.is_integer_dtype(dtype) and series.nunique() < 20:
            return "categorical_numeric"
        return "numeric"
    
    # Check if string/object
    if pd.api.types.is_string_dtype(dtype) or dtype == 'object':
        return "categorical"
    
    return "unknown"


def get_feature_engineering_method(feature_name: str) -> Dict[str, str]:
    """
    Determine how this feature is processed in feature engineering.
    确定该特征在特征工程中的处理方法。
    
    Returns:
        Dict with 'baseline_method' and 'solution_method'
    """
    baseline_method = "unknown"
    solution_method = "unknown"
    
    # Engineered features (created in feature engineering)
    engineered_features = {
        'follower_following_ratio': {
            'baseline': 'Created: user.followers_count / (user.friends_count + 1e-6)',
            'solution': 'Created: user.followers_count / (user.friends_count + 1e-6)'
        },
        'follower_following_diff': {
            'baseline': 'Created: user.followers_count - user.friends_count',
            'solution': 'Created: user.followers_count - user.friends_count'
        },
        'account_age_days': {
            'baseline': 'Created: calculate_account_age(user.created_at)',
            'solution': 'Created: calculate_account_age(user.created_at)'
        },
        'tweets_per_day': {
            'baseline': 'Created: user.statuses_count / (account_age_days + 1e-6)',
            'solution': 'Created: user.statuses_count / (account_age_days + 1e-6)'
        },
        'listed_per_day': {
            'baseline': 'Created: user.listed_count / (account_age_days + 1e-6)',
            'solution': 'Created: user.listed_count / (account_age_days + 1e-6)'
        },
        'tweet_engagement_rate': {
            'baseline': 'Created: (retweet_count + favorite_count) / (followers_count + 1e-6)',
            'solution': 'Created: (retweet_count + favorite_count) / (followers_count + 1e-6)'
        },
        'is_reply': {
            'baseline': 'Created: 1 if in_reply_to_user_id not null, else 0',
            'solution': 'Created: 1 if in_reply_to_user_id not null, else 0'
        },
        'mentions_count': {
            'baseline': 'Created: len(entities.user_mentions)',
            'solution': 'Created: len(entities.user_mentions)'
        },
        'hashtags_count': {
            'baseline': 'Created: len(entities.hashtags)',
            'solution': 'Created: len(entities.hashtags)'
        },
        'urls_count': {
            'baseline': 'Created: len(entities.urls)',
            'solution': 'Created: len(entities.urls)'
        },
        'source_encoded': {
            'baseline': 'OrdinalEncoder on source column, then drop source',
            'solution': 'OrdinalEncoder on source column, then drop source'
        }
    }
    
    if feature_name in engineered_features:
        baseline_method = engineered_features[feature_name]['baseline']
        solution_method = engineered_features[feature_name]['solution']
    else:
        # Check if it's a raw feature
        # In baseline: 
        #   - Numeric features: used as-is
        #   - Non-numeric features: LabelEncoder (from test_xgboost.py pattern)
        # In solution (data_processing.py):
        #   - Numeric features: used as-is
        #   - Non-numeric features: OrdinalEncoder
        
        # For baseline (test_xgboost.py approach):
        #   - Numeric features: used as-is (with median imputation)
        #   - Non-numeric features: LabelEncoder
        # For solution (data_processing.py engineer_features):
        #   - Most raw features are dropped during feature engineering
        #   - Only selected features are kept and used as-is or encoded
        
        # Since we're now using the baseline approach (all tabular features),
        # we need to check if this feature would be kept in solution
        baseline_method = "Used as-is (numeric feature with median imputation). If originally categorical, LabelEncoder was applied (baseline approach)."
        solution_method = "May be dropped during feature engineering. If kept, used as-is or encoded with OrdinalEncoder (solution approach)."
    
    return {
        "baseline_method": baseline_method,
        "solution_method": solution_method
    }


def calculate_mutual_information(series: pd.Series, target: pd.Series) -> Optional[float]:
    """
    Calculate mutual information between feature and target.
    计算特征与目标变量的互信息。
    """
    try:
        # Align series and target
        aligned_data = pd.DataFrame({
            'feature': series,
            'target': target
        }).dropna()
        
        if len(aligned_data) < 10:  # Need minimum samples
            return None
        
        # For categorical features, use discrete MI
        if not pd.api.types.is_numeric_dtype(aligned_data['feature']):
            # Convert to numeric codes for MI calculation
            feature_codes = pd.Categorical(aligned_data['feature']).codes
            mi_score = mutual_info_classif(
                feature_codes.reshape(-1, 1),
                aligned_data['target'].values,
                discrete_features=True,
                random_state=42
            )[0]
        else:
            # For numeric features
            mi_score = mutual_info_classif(
                aligned_data[['feature']].values,
                aligned_data['target'].values,
                discrete_features=False,
                random_state=42
            )[0]
        
        return float(mi_score) if not np.isnan(mi_score) else None
    except Exception as e:
        logger.debug(f"  Could not calculate MI: {e}")
        return None


def calculate_skewness_kurtosis(series: pd.Series) -> Dict[str, Optional[float]]:
    """
    Calculate skewness and kurtosis for numeric features.
    计算数值特征的偏度和峰度。
    """
    result = {"skewness": None, "kurtosis": None}
    
    if not pd.api.types.is_numeric_dtype(series):
        return result
    
    non_null_series = series.dropna()
    if len(non_null_series) < 3:  # Need minimum samples
        return result
    
    try:
        result["skewness"] = float(stats.skew(non_null_series))
        result["kurtosis"] = float(stats.kurtosis(non_null_series))
    except Exception as e:
        logger.debug(f"  Could not calculate skewness/kurtosis: {e}")
    
    return result


def calculate_zero_inflation_rate(series: pd.Series) -> Optional[float]:
    """
    Calculate zero-inflation rate for numeric features.
    计算数值特征的零值膨胀率。
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None
    
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return None
    
    zero_count = (non_null_series == 0).sum()
    return float(zero_count / len(non_null_series) * 100)


def calculate_conditional_probability_diff(series: pd.Series, target: pd.Series, value: Any) -> Optional[float]:
    """
    Calculate conditional probability difference: P(Y=1|X=x) - P(Y=1)
    计算条件概率差异。
    """
    try:
        aligned_data = pd.DataFrame({
            'feature': series,
            'target': target
        }).dropna()
        
        if len(aligned_data) == 0:
            return None
        
        # Global probability P(Y=1)
        global_prob = aligned_data['target'].mean()
        
        # Conditional probability P(Y=1|X=x)
        mask = aligned_data['feature'] == value
        if mask.sum() == 0:
            return None
        
        conditional_prob = aligned_data.loc[mask, 'target'].mean()
        
        return float(conditional_prob - global_prob)
    except Exception:
        return None


def calculate_id_purity(series: pd.Series, target: pd.Series, min_count: int = 5) -> Optional[Dict[str, Any]]:
    """
    Calculate ID purity (Gini impurity) for high-cardinality ID features.
    计算ID特征的纯度（基尼不纯度）。
    """
    try:
        aligned_data = pd.DataFrame({
            'feature': series,
            'target': target
        }).dropna()
        
        if len(aligned_data) == 0:
            return None
        
        # Count occurrences of each ID
        id_counts = aligned_data['feature'].value_counts()
        frequent_ids = id_counts[id_counts >= min_count].index
        
        if len(frequent_ids) == 0:
            return None
        
        purities = []
        for id_val in frequent_ids:
            id_targets = aligned_data[aligned_data['feature'] == id_val]['target']
            if len(id_targets) == 0:
                continue
            
            # Purity: 1 - Gini impurity
            # Gini = 1 - sum(p_i^2) where p_i is proportion of class i
            value_counts = id_targets.value_counts(normalize=True)
            gini_impurity = 1 - (value_counts ** 2).sum()
            purity = 1 - gini_impurity
            purities.append(purity)
        
        if len(purities) == 0:
            return None
        
        return {
            "mean_purity": float(np.mean(purities)),
            "median_purity": float(np.median(purities)),
            "min_purity": float(np.min(purities)),
            "max_purity": float(np.max(purities)),
            "frequent_ids_count": int(len(frequent_ids))
        }
    except Exception as e:
        logger.debug(f"  Could not calculate ID purity: {e}")
        return None


def calculate_temporal_statistics(series: pd.Series, target: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Calculate temporal dynamics statistics (hour distribution, weekday/weekend ratio).
    计算时间动力学统计（小时分布、周中/周末比率）。
    """
    try:
        # Try to parse as datetime
        aligned_data = pd.DataFrame({
            'datetime_str': series,
            'target': target
        }).dropna()
        
        if len(aligned_data) == 0:
            return None
        
        # Parse datetime
        try:
            # Use faster parsing if format is standard, or coerce for mixed
            # Twitter format: "Wed Mar 31 18:37:10 +0000 2021"
            # %a %b %d %H:%M:%S %z %Y
            twitter_fmt = "%a %b %d %H:%M:%S %z %Y"
            try:
                aligned_data['datetime'] = pd.to_datetime(aligned_data['datetime_str'], format=twitter_fmt, errors='coerce')
            except:
                aligned_data['datetime'] = pd.to_datetime(aligned_data['datetime_str'], errors='coerce')
            
            aligned_data = aligned_data.dropna(subset=['datetime'])
        except:
            return None
        
        if len(aligned_data) == 0:
            return None
        
        # Extract hour and weekday
        aligned_data['hour'] = aligned_data['datetime'].dt.hour
        aligned_data['weekday'] = aligned_data['datetime'].dt.weekday  # 0=Monday, 6=Sunday
        aligned_data['is_weekend'] = aligned_data['weekday'].isin([5, 6])
        
        # Hour distribution KL divergence
        influencer_hours = aligned_data[aligned_data['target'] == 1]['hour'].value_counts(normalize=True).sort_index()
        observer_hours = aligned_data[aligned_data['target'] == 0]['hour'].value_counts(normalize=True).sort_index()
        
        # Align indices for KL divergence
        all_hours = sorted(set(influencer_hours.index) | set(observer_hours.index))
        p = np.array([influencer_hours.get(h, 0) for h in all_hours])
        q = np.array([observer_hours.get(h, 0) for h in all_hours])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()
        
        kl_divergence = float(np.sum(p * np.log(p / q)))
        
        # Weekday/Weekend ratio
        influencer_weekend_ratio = aligned_data[aligned_data['target'] == 1]['is_weekend'].mean()
        observer_weekend_ratio = aligned_data[aligned_data['target'] == 0]['is_weekend'].mean()
        
        return {
            "hour_kl_divergence": kl_divergence,
            "influencer_weekend_ratio": float(influencer_weekend_ratio),
            "observer_weekend_ratio": float(observer_weekend_ratio),
            "weekend_ratio_diff": float(influencer_weekend_ratio - observer_weekend_ratio)
        }
    except Exception as e:
        logger.debug(f"  Could not calculate temporal statistics: {e}")
        return None


def calculate_statistics(df: pd.DataFrame, target: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
    """
    Calculate comprehensive statistics for all features.
    计算所有特征的全面统计信息。
    
    Args:
        df: Feature DataFrame
        target: Target variable (for relationship metrics)
    """
    results = []
    
    for col_idx, col in enumerate(df.columns):
        if col_idx % 10 == 0:
            logger.info(f"Processing feature {col_idx+1}/{len(df.columns)}: {col}")
        series = df[col]
        
        # Basic info
        missing_count = int(series.isna().sum())
        total_count = len(series)
        missing_pct = float(missing_count / total_count * 100) if total_count > 0 else 0.0
        
        # Feature type
        feature_type = detect_feature_type(series)
        
        # Top values (for all types)
        # Skip top_values for complex types (lists/arrays) to save time
        is_complex_type = False
        try:
            sample_non_null = series.dropna().head(10)
            if len(sample_non_null) > 0:
                first_val = sample_non_null.iloc[0]
                if isinstance(first_val, (list, dict, np.ndarray)):
                    is_complex_type = True
        except:
            pass
        
        if is_complex_type:
            top_values = []
            logger.debug(f"  Skipping top_values for complex type column: {col}")
        else:
            try:
                top_values = get_top_values(series, top_n=5)
            except Exception as e:
                logger.debug(f"  Warning: Could not get top values for {col}: {e}")
                top_values = []
        
        # Statistics for numeric features
        stats_dict = {}
        mode_value = None
        
        if pd.api.types.is_numeric_dtype(series):
            non_null_series = series.dropna()
            
            if len(non_null_series) > 0:
                # Convert boolean to int for calculations
                if pd.api.types.is_bool_dtype(non_null_series):
                    non_null_series = non_null_series.astype(int)
                
                try:
                    stats_dict = {
                        "mean": float(non_null_series.mean()) if len(non_null_series) > 0 else None,
                        "median": float(non_null_series.median()) if len(non_null_series) > 0 else None,
                        "q25": float(non_null_series.quantile(0.25)) if len(non_null_series) > 0 else None,
                        "q75": float(non_null_series.quantile(0.75)) if len(non_null_series) > 0 else None,
                        "std": float(non_null_series.std()) if len(non_null_series) > 0 else None,
                        "min": float(non_null_series.min()) if len(non_null_series) > 0 else None,
                        "max": float(non_null_series.max()) if len(non_null_series) > 0 else None,
                    }
                except (TypeError, ValueError) as e:
                    logger.warning(f"  Warning: Could not calculate quantiles for {col}: {e}")
                    stats_dict = {
                        "mean": float(non_null_series.mean()) if len(non_null_series) > 0 else None,
                        "median": float(non_null_series.median()) if len(non_null_series) > 0 else None,
                        "std": float(non_null_series.std()) if len(non_null_series) > 0 else None,
                        "min": float(non_null_series.min()) if len(non_null_series) > 0 else None,
                        "max": float(non_null_series.max()) if len(non_null_series) > 0 else None,
                    }
                
                # Mode (most frequent value)
                mode_value = get_mode(non_null_series)
                if mode_value is not None:
                    if isinstance(mode_value, (np.integer, np.floating)):
                        stats_dict["mode"] = mode_value.item()
                    else:
                        stats_dict["mode"] = mode_value
                
                # Add skewness and kurtosis
                skew_kurt = calculate_skewness_kurtosis(series)
                stats_dict.update(skew_kurt)
                
                # Add zero-inflation rate
                zero_inflation = calculate_zero_inflation_rate(series)
                if zero_inflation is not None:
                    stats_dict["zero_inflation_rate"] = zero_inflation
        
        # Relationship metrics (if target is provided)
        relationship_metrics = {}
        if target is not None and not is_complex_type:
            # Mutual Information (skip for complex types)
            try:
                mi_score = calculate_mutual_information(series, target)
                if mi_score is not None:
                    relationship_metrics["mutual_information"] = mi_score
            except Exception as e:
                logger.debug(f"  Could not calculate MI for {col}: {e}")
            
            # Conditional probability differences for top values
            if top_values and len(top_values) > 0:
                cond_probs = []
                for tv in top_values[:3]:  # Top 3 values
                    try:
                        # Try to parse value back to original type
                        val = tv['value']
                        if val == "NaN":
                            continue
                        # For numeric, try to convert
                        if pd.api.types.is_numeric_dtype(series):
                            try:
                                val = float(val)
                            except:
                                pass
                        
                        cond_prob_diff = calculate_conditional_probability_diff(series, target, val)
                        if cond_prob_diff is not None:
                            cond_probs.append({
                                "value": val,
                                "conditional_probability_diff": cond_prob_diff
                            })
                    except:
                        pass
                if cond_probs:
                    relationship_metrics["top_values_conditional_prob"] = cond_probs
            
            # ID Purity (for high-cardinality features)
            # Check if this is likely an ID feature (high cardinality, not complex type)
            try:
                nunique = series.nunique()
                if nunique > 100 and not is_complex_type:  # Likely an ID feature
                    id_purity = calculate_id_purity(series, target)
                    if id_purity:
                        relationship_metrics["id_purity"] = id_purity
            except (TypeError, ValueError):
                # Skip for unhashable types (lists, dicts, etc.)
                pass
            
            # Temporal statistics (for datetime features)
            # Only calculate for actual datetime columns (not all columns)
            if ('created_at' in col.lower() or 'timestamp' in col.lower()) and not is_complex_type:
                try:
                    # Quick check: try to parse first non-null value
                    sample = series.dropna().head(1)
                    if len(sample) > 0:
                        test_val = sample.iloc[0]
                        # Check if it looks like a datetime string
                        if isinstance(test_val, str) and ('+' in test_val or '-' in test_val or ':' in test_val):
                            temporal_stats = calculate_temporal_statistics(series, target)
                            if temporal_stats:
                                relationship_metrics["temporal_dynamics"] = temporal_stats
                except:
                    pass
        
        # Feature engineering methods
        engineering_methods = get_feature_engineering_method(col)
        
        # Build result dictionary
        result = {
            "feature_name": col,
            "feature_type": feature_type,
            "total_count": total_count,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "top_values": top_values,
            "statistics": stats_dict,
            "relationship_metrics": relationship_metrics if relationship_metrics else None,
            "xgboost_baseline_method": engineering_methods["baseline_method"],
            "xgboost_solution_method": engineering_methods["solution_method"]
        }
        
        results.append(result)
    
    return results


def main():
    """
    Main function to generate feature statistics report.
    生成特征统计报表的主函数。
    
    Generates statistics for ALL tabular features using RAW, UNPROCESSED values.
    No missing value imputation, no encoding - just original data statistics.
    """
    logger.info("="*60)
    logger.info("Generating Feature Statistics Report (RAW VALUES)")
    logger.info("="*60)
    
    # Load training data
    logger.info("\n[1] Loading training data...")
    train_file = str(DATA_DIR / 'train.jsonl')
    train_df = dp.load_data(train_file, clean_text=True)
    logger.info(f"  Loaded {len(train_df)} records with {len(train_df.columns)} columns")
    
    # Extract tabular features (like test_xgboost.py - exclude text columns only)
    logger.info("\n[2] Extracting tabular features (excluding text columns)...")
    exclude_cols = ['text', 'extended_tweet.full_text', 'label', 'challenge_id']
    table_cols = [col for col in train_df.columns if col not in exclude_cols]
    X_train = train_df[table_cols].copy()
    
    # Extract target variable for relationship metrics
    if 'label' in train_df.columns:
        y_train = train_df['label']
        logger.info(f"  Extracted {len(table_cols)} tabular features (excluding text columns)")
        logger.info(f"  Target variable found: {y_train.value_counts().to_dict()}")
    else:
        y_train = None
        logger.warning("  No target variable (label) found - relationship metrics will be skipped")
    
    # Calculate statistics on RAW, UNPROCESSED data
    # No missing value imputation, no encoding - just original values
    logger.info("\n[3] Calculating statistics on RAW (unprocessed) values...")
    logger.info("  Note: Statistics are computed on original data before any preprocessing")
    logger.info("  Including relationship metrics: Mutual Information, Skewness, ID Purity, etc.")
    statistics = calculate_statistics(X_train, target=y_train)
    logger.info(f"  Processed {len(statistics)} features")
    
    # Create output directory
    output_dir = get_preprocessing_dir('visualizations/stats')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON using NumpyEncoder
    output_file = output_dir / 'feature_statistics_report.json'
    logger.info(f"\n[6] Saving statistics to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    logger.info(f"  ✓ Statistics saved successfully!")
    logger.info(f"\n{'='*60}")
    logger.info("Feature Statistics Report Generation Complete!")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Total features: {len(statistics)}")
    logger.info(f"  (This matches the '194 tabular features' approach)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()

