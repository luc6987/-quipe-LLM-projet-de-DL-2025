"""
Hyperparameter Optimization Module
 / Module d'optimisation des hyperparamètres / Hyperparameter Optimization Module / Hyperparameter-Optimierungsmodul

Implements Phase 3: Bayesian Optimization using Optuna with GPU acceleration.
OptunaGPU / Implements Phase 3: Bayesian Optimization using Optuna with GPU acceleration

Bayesian Optimization /  / Optimisation bayésienne / Bayes'sche Optimierung:
- Uses TPE (Tree-structured Parzen Estimator) algorithm
- Builds probabilistic model of objective function
- Updates posterior probability distribution after each trial
- More efficient than grid/random search

Etymology: "Bayesian" from Thomas Bayes (1701-1761), who formulated Bayes' theorem
for updating probability distributions based on new evidence.
"""

import os
# Fix for Python 3.9 locale encoding issue with Optuna/Alembic
# Set encoding environment variable early to avoid LookupError: unknown encoding: locale
if 'PYTHONIOENCODING' not in os.environ:
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 
from src._imports import (
    np, pd, xgb, Dict, Any, Tuple, Optional, logging,
    json, Path,
    sklearn_accuracy_score as accuracy_score,
    sklearn_GroupKFold as GroupKFold,
    sklearn_StratifiedKFold as StratifiedKFold,
    OPTUNA_AVAILABLE, optuna
)
import tempfile
import time
import hashlib
import locale

# Import unified path configuration
from src.utils.paths import get_optuna_cache_dir

logger = logging.getLogger(__name__)


def _get_cache_dir() -> Path:
    """
    Get cache directory path. / 
    
    Returns:
        Path to Optuna cache directory / Path to Optuna cache directory
    """
    return get_optuna_cache_dir()


def _generate_study_id(use_kaggle_score: bool, competition: str, 
                       cv_folds: int, use_gpu: bool, cv_strategy: str = 'stratified') -> str:
    """
    Generate unique study ID based on configuration. / study ID
    
    Args:
        use_kaggle_score: Whether using Kaggle score / Whether using Kaggle score
        competition: Competition name / Competition name
        cv_folds: Number of CV folds / Number of CV folds
        use_gpu: Whether using GPU / Whether using GPU
        cv_strategy: CV strategy ('stratified' or 'group')
        
    Returns:
        Unique study ID string / Unique study ID string
    """
    config_str = f"kaggle_{competition}_{cv_folds}folds_gpu{use_gpu}_{cv_strategy}" if use_kaggle_score else f"cv_{cv_folds}folds_gpu{use_gpu}_{cv_strategy}"
    # Create hash for uniqueness
    study_id = hashlib.md5(config_str.encode()).hexdigest()[:12]
    return f"optuna_study_{study_id}"


def _save_optimization_config(cache_dir: Path, study_id: str, config: Dict[str, Any]) -> None:
    """
    Save optimization configuration to JSON file. / JSON
    
    Args:
        cache_dir: Cache directory path / Cache directory path
        study_id: Study ID / Study ID
        config: Configuration dictionary / Configuration dictionary
    """
    config_file = cache_dir / f"{study_id}_config.json"
    
    # Convert numpy types to Python native types for JSON serialization
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, (np.integer, np.floating)):
            config_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            config_serializable[key] = value.tolist()
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            config_serializable[key] = str(value.shape)  # Just save shape info
        else:
            config_serializable[key] = value
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Saved optimization config to: {config_file}")


def _load_optimization_config(cache_dir: Path, study_id: str) -> Optional[Dict[str, Any]]:
    """
    Load optimization configuration from JSON file.
    JSON
    
    Args:
        cache_dir: Cache directory path
        study_id: Study ID
        
    Returns:
        Configuration dictionary or None if not found
    """
    config_file = cache_dir / f"{study_id}_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"  Loaded optimization config from: {config_file}")
            return config
        except Exception as e:
            logger.warning(f"  Could not load config file {config_file}: {e}")
    return None


def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
                             n_trials: int = 100, cv_folds: int = 5,
                             use_gpu: bool = True, gpu_params: Dict[str, Any] = None,
                             timeout: Optional[int] = None, verbose: bool = True,
                             use_kaggle_score: bool = False, X_test: Optional[pd.DataFrame] = None,
                             test_ids: Optional[pd.Series] = None,
                             competition: str = 'influencers-or-observers-predicting-social-roles',
                             cv_strategy: str = 'stratified') -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna Bayesian Optimization.
    OptunaXGBoost
    
    Can use either CV score or Kaggle submission score as optimization target.
    CVKaggle
    
    Search Space (regularized to reduce overfitting):
    - learning_rate: log-uniform [1e-3, 0.1]
    - max_depth: int [3, 7]               (was [3, 10])
    - subsample: float [0.6, 1.0]         (was [0.5, 1.0])
    - colsample_bytree: float [0.6, 1.0]  (was [0.5, 1.0])
    - min_child_weight: int [3, 20]       (was [1, 20])
    - gamma: float [0.0, 5.0]
    - reg_alpha: float [0.0, 2.0]
    - reg_lambda: float [0.0, 5.0]
    
    Baseline Parameters:
    - scale_pos_weight: 1.127 (53% negative / 47% positive)
    - tree_method: 'hist' with device='cuda' (GPU acceleration)
    - objective: 'binary:logistic'
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels Series
        groups: User ID Series for GroupKFold
        n_trials: Number of Optuna optimization trials
        cv_folds: Number of CV folds (default: 5)
        use_gpu: Whether to use GPU acceleration
        gpu_params: GPU parameters dict from utils.detect_gpu_for_xgboost()
        timeout: Optimization timeout in seconds (None = no limit)
        verbose: Whether to print progress
        use_kaggle_score: If True, use Kaggle submission score instead of CV score
        X_test: Test features DataFrame (required if use_kaggle_score=True)
        test_ids: Test IDs Series (required if use_kaggle_score=True)
        competition: Kaggle competition name (required if use_kaggle_score=True)
        cv_strategy: 'stratified' or 'group' (default: 'stratified')
        
    Returns:
        Dictionary containing:
        - 'best_params': Best hyperparameters found
        - 'best_score': Best score (CV or Kaggle)
        - 'study': Optuna study object
        - 'best_iterations': Mean best iterations from CV
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Please install with: pip install optuna"
        )
    
    logger.info("="*60)
    logger.info("Hyperparameter Optimization: Optuna Bayesian Optimization")
    logger.info("="*60)
    logger.info(f"  Optimization trials: {n_trials}")
    logger.info(f"  Optimization target: {'Kaggle Score' if use_kaggle_score else 'CV Score'}")
    if timeout:
        logger.info(f"  Timeout: {timeout} seconds")
    
    # Validate Kaggle score mode requirements
    if use_kaggle_score:
        if X_test is None or test_ids is None:
            raise ValueError("X_test and test_ids are required when use_kaggle_score=True")
        
        logger.warning("  ⚠ WARNING: Using Kaggle score for optimization")
        logger.warning("    - Each trial will submit to Kaggle (consumes submission quota)")
        logger.warning(f"    - {n_trials} trials = {n_trials} Kaggle submissions")
        logger.warning("    - Each submission may take 1-5 minutes to process")
        logger.warning("    - Total time: ~{} minutes minimum".format(n_trials * 2))
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()
            logger.info("  ✓ Kaggle API authenticated")
        except ImportError:
            raise ImportError("Kaggle API not installed. Install with: pip install kaggle")
        except Exception as e:
            raise RuntimeError(f"Kaggle API authentication failed: {e}")
    
    # Convert to numpy arrays
    if isinstance(X_train, pd.DataFrame):
        X_array = X_train.values
    else:
        X_array = X_train
    
    if isinstance(y_train, pd.Series):
        y_array = y_train.values
    else:
        y_array = y_train
    
    if isinstance(groups, pd.Series):
        groups_array = groups.values
    else:
        groups_array = groups
    
    # Calculate scale_pos_weight for class imbalance
    negative_count = np.sum(y_array == 0)
    positive_count = np.sum(y_array == 1)
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.127
    logger.info(f"  Class distribution: {negative_count} negative, {positive_count} positive")
    logger.info(f"  scale_pos_weight: {scale_pos_weight:.3f}")
    
    # Initialize CV
    if cv_strategy == 'stratified':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        logger.info(f"  Using StratifiedKFold (CV folds: {cv_folds})")
    else:
        cv = GroupKFold(n_splits=cv_folds)
        logger.info(f"  Using GroupKFold (CV folds: {cv_folds})")
    
    # Prepare GPU parameters
    if use_gpu and gpu_params:
        base_gpu_params = gpu_params.copy()
        logger.info(f"  GPU acceleration: {gpu_params}")
    else:
        base_gpu_params = {'tree_method': 'hist'}
        logger.info("  Using CPU mode")
    
    # ========== Setup Study Persistence ==========
    cache_dir = _get_cache_dir()
    study_id = _generate_study_id(use_kaggle_score, competition, cv_folds, use_gpu, cv_strategy)
    study_db_path = cache_dir / f"{study_id}.db"
    study_name = f"xgboost_optimization_{study_id}"
    
    # Prepare configuration to save
    optimization_config = {
        'study_id': study_id,
        'cv_strategy': cv_strategy,
        'use_kaggle_score': use_kaggle_score,
        'competition': competition,
        'cv_folds': cv_folds,
        'use_gpu': use_gpu,
        'gpu_params': base_gpu_params,
        'scale_pos_weight': float(scale_pos_weight),
        'n_trials': n_trials,
        'timeout': timeout,
        'data_shape': {
            'n_samples': len(X_array),
            'n_features': X_array.shape[1] if len(X_array.shape) > 1 else 1,
            'n_classes': len(np.unique(y_array)),
            'class_distribution': {
                'negative': int(negative_count),
                'positive': int(positive_count)
            }
        }
    }
    
    # Try to load previous study
    study = None
    previous_trials = 0
    if study_db_path.exists():
        try:
            logger.info(f"\n  Found previous study database: {study_db_path}")
            logger.info(f"  Attempting to load previous optimization progress...")
            
            # Fix for Python 3.9 locale encoding issue
            try:
                current_locale = locale.getlocale()
                if current_locale[1] is None:
                    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except (locale.Error, AttributeError):
                # os is already imported at module level
                if 'PYTHONIOENCODING' not in os.environ:
                    os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # Load study from database
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=f"sqlite:///{study_db_path}"
                )
                previous_trials = len(study.trials)
                logger.info(f"  ✓ Loaded previous study with {previous_trials} completed trials")
                logger.info(f"  Best previous score: {study.best_value:.6f}")
                logger.info(f"  Best previous params: {study.best_params}")
            except (LookupError, UnicodeDecodeError) as e:
                if 'locale' in str(e).lower() or 'encoding' in str(e).lower():
                    logger.warning(f"  ⚠ Could not load study due to encoding issue: {e}")
                    logger.info(f"  Creating new study instead")
                    logger.info(f"  To fix: set environment variable PYTHONIOENCODING=utf-8")
                    study = None
                    previous_trials = 0
                else:
                    raise
            
            # Load and verify config
            saved_config = _load_optimization_config(cache_dir, study_id)
            if saved_config:
                # Verify config matches (basic check)
                if saved_config.get('use_kaggle_score') != use_kaggle_score:
                    logger.warning(f"  ⚠ Config mismatch: use_kaggle_score changed")
                    logger.warning(f"    Previous: {saved_config.get('use_kaggle_score')}, Current: {use_kaggle_score}")
                    logger.warning(f"    Creating new study...")
                    study = None
                    previous_trials = 0  # Reset when creating new study
        except Exception as e:
            logger.warning(f"  Could not load previous study: {e}")
            logger.info(f"  Creating new study...")
            study = None
            previous_trials = 0  # Reset when creating new study
    
    # Create new study if not loaded
    if study is None:
        logger.info(f"\n  Creating new optimization study: {study_name}")
        logger.info(f"  Study database: {study_db_path}")
        
        # Fix for Python 3.9 locale encoding issue
        # Set locale encoding to UTF-8 if locale encoding is not available
        try:
            # Try to get locale encoding
            current_locale = locale.getlocale()
            if current_locale[1] is None:
                # Set to UTF-8 if locale encoding is None
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except (locale.Error, AttributeError):
            # If locale setting fails, try to set environment variable
            # os is already imported at module level
            # Set PYTHONIOENCODING to UTF-8 as fallback
            if 'PYTHONIOENCODING' not in os.environ:
                os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Create study with SQLite storage for persistence
        # Use try-except to handle encoding issues gracefully
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{study_db_path}",
                direction='maximize',  # Maximize accuracy
                sampler=optuna.samplers.TPESampler(seed=42),
                load_if_exists=True  # Load if exists (shouldn't happen, but safety)
            )
        except (LookupError, UnicodeDecodeError) as e:
            if 'locale' in str(e).lower() or 'encoding' in str(e).lower():
                logger.warning(f"  ⚠ SQLite storage encoding issue detected: {e}")
                logger.info(f"  Falling back to in-memory storage (progress will not persist)")
                logger.info(f"  To fix: set environment variable PYTHONIOENCODING=utf-8")
                # Fallback to in-memory storage
                study = optuna.create_study(
                    study_name=study_name,
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
            else:
                raise
        
        # Save configuration
        _save_optimization_config(cache_dir, study_id, optimization_config)
        logger.info(f"  ✓ New study created and config saved")
    else:
        logger.info(f"  Continuing optimization from trial {previous_trials + 1}")
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function: maximize score (CV or Kaggle).
        OptunaCVKaggle
        
        Uses TPE (Tree-structured Parzen Estimator) sampler:
        - Models p(x|good) and p(x|bad) distributions
        - Samples from p(x|good) / p(x|bad) ratio
        - More efficient exploration of search space
        """
        # Define search space (exact as specified)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            
            # Hyperparameters to optimize
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            
            # Fixed parameters
            'n_estimators': 800,  # Reduced cap, use early stopping to pick best_iteration
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        # Add GPU parameters
        params.update(base_gpu_params)
        
        if use_kaggle_score:
            # ========== Use Kaggle Score Mode ==========
            # Train on full training set and submit to Kaggle
            logger.info(f"  Trial {trial.number}: Training model for Kaggle submission...")
            
            # Determine n_estimators from CV (quick CV to get best_iteration)
            cv_best_iterations = []
            
            # Setup splitter generator based on strategy
            if cv_strategy == 'stratified':
                splitter = cv.split(X_array, y_array)
            else:
                splitter = cv.split(X_array, y_array, groups=groups_array)
                
            for train_idx, val_idx in splitter:
                X_fold_train = X_array[train_idx]
                X_fold_val = X_array[val_idx]
                y_fold_train = y_array[train_idx]
                y_fold_val = y_array[val_idx]
                
                model_params_cv = params.copy()
                model_params_cv['early_stopping_rounds'] = 50
                model_cv = xgb.XGBClassifier(**model_params_cv)
                model_cv.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
                
                if hasattr(model_cv, 'best_iteration') and model_cv.best_iteration is not None:
                    cv_best_iterations.append(model_cv.best_iteration)
                else:
                    cv_best_iterations.append(params['n_estimators'])
            
            mean_best_iter = int(np.mean(cv_best_iterations) * 1.1)  # 1.1x multiplier
            params['n_estimators'] = mean_best_iter
            
            # Train final model on full training set
            logger.info(f"  Trial {trial.number}: Training final model (n_estimators={mean_best_iter})...")
            final_model = xgb.XGBClassifier(**params)
            final_model.fit(X_array, y_array, verbose=False)
            
            # Generate predictions
            if isinstance(X_test, pd.DataFrame):
                X_test_array = X_test.values.astype(np.float32)
            else:
                X_test_array = X_test.astype(np.float32)
            
            predictions = final_model.predict(X_test_array)
            
            # Save submission temporarily
            # Import utils function (handle both relative and absolute imports)
            try:
                from utils import save_submission
            except ImportError:
                import sys
                solution_dir = os.path.dirname(os.path.abspath(__file__))
                if solution_dir not in sys.path:
                    sys.path.insert(0, solution_dir)
                from utils import save_submission
            
            temp_dir = tempfile.mkdtemp()
            temp_submission = os.path.join(temp_dir, f'submission_trial_{trial.number}.csv')
            
            save_submission(predictions, test_ids, temp_submission)
            
            # Submit to Kaggle
            logger.info(f"  Trial {trial.number}: Submitting to Kaggle...")
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                
                kaggle_api = KaggleApi()
                kaggle_api.authenticate()
                
                submission_message = f"Optuna Trial {trial.number}: lr={params['learning_rate']:.4f}, depth={params['max_depth']}, subsample={params['subsample']:.3f}"
                
                kaggle_api.competition_submit(
                    file_name=temp_submission,
                    message=submission_message,
                    competition=competition
                )
                
                # Force output flush to ensure message is displayed immediately
                import sys
                sys.stdout.flush()
                
                # Wait for submission to process (Kaggle usually takes 1-2 minutes)
                logger.info(f"  Trial {trial.number}: ✓ Submission uploaded successfully!")
                logger.info(f"  Trial {trial.number}: Waiting for Kaggle score (this may take 1-5 minutes)...")
                logger.info(f"  Trial {trial.number}: Check status at: https://www.kaggle.com/competitions/{competition}/submissions")
                sys.stdout.flush()
                
                max_wait_time = 300  # 5 minutes max wait (can be configured)
                wait_interval = 15  # Check every 15 seconds
                elapsed_time = 0
                score = None
                submission_ref = None
                
                # Get submission reference for tracking
                try:
                    time.sleep(2)  # Wait a bit for submission to appear in API
                    submissions = kaggle_api.competition_submissions(competition)
                    if submissions:
                        latest = submissions[0]
                        submission_ref = getattr(latest, 'ref', None) or getattr(latest, 'fileName', None) or getattr(latest, 'description', None)
                        status = getattr(latest, 'status', 'unknown')
                        logger.info(f"  Trial {trial.number}: Submission found, status: {status}")
                except Exception as e:
                    logger.debug(f"  Trial {trial.number}: Could not get initial submission info: {e}")
                
                # Poll for score
                check_count = 0
                logger.info(f"  Trial {trial.number}: Starting to poll for score (checking every {wait_interval}s)...")
                import sys
                sys.stdout.flush()
                
                while elapsed_time < max_wait_time:
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
                    check_count += 1
                    
                    try:
                        submissions = kaggle_api.competition_submissions(competition)
                        if submissions:
                            # Find our submission (by description matching trial number)
                            our_submission = None
                            for sub in submissions:
                                desc = getattr(sub, 'description', '') or getattr(sub, 'fileName', '')
                                if f'Trial {trial.number}' in str(desc) or submission_ref:
                                    our_submission = sub
                                    break
                            
                            # If not found by description, use latest
                            if our_submission is None:
                                our_submission = submissions[0]
                            
                            # Get status
                            status = getattr(our_submission, 'status', 'unknown')
                            
                            # Try to get score (check all possible attribute names)
                            score = None
                            for attr_name in ['publicScore', 'score', 'publicScoreFloat', 'scoreFloat', 
                                            'publicScoreString', 'scoreString', 'publicScoreValue']:
                                score_attr = getattr(our_submission, attr_name, None)
                                if score_attr is not None:
                                    try:
                                        score = float(score_attr)
                                        break
                                    except (ValueError, TypeError):
                                        continue
                            
                            # Log progress every 30 seconds
                            if check_count % 2 == 0:  # Every 30 seconds (2 * 15s)
                                logger.info(f"  Trial {trial.number}: [{elapsed_time}s] Status={status}, Score={'available' if score is not None else 'pending'}...")
                                import sys
                                sys.stdout.flush()
                            
                            if score is not None:
                                logger.info(f"  ✓ Trial {trial.number}: Kaggle score received: {score:.6f}")
                                import sys
                                sys.stdout.flush()
                                break
                            elif status in ['complete', 'done']:
                                # Status is complete but score not available - might need more time
                                logger.debug(f"  Trial {trial.number}: Status is {status} but score not yet available")
                        else:
                            logger.debug(f"  Trial {trial.number}: No submissions found yet")
                    except Exception as e:
                        logger.debug(f"  Trial {trial.number}: Error checking submission status: {e}")
                        # Continue waiting
                
                if score is None:
                    logger.warning(f"  ⚠ Trial {trial.number}: Kaggle score not available after {max_wait_time}s")
                    logger.warning(f"  Trial {trial.number}: Submission was successful, but score is still processing")
                    logger.warning(f"  Trial {trial.number}: Falling back to CV score for this trial")
                    logger.info(f"  Trial {trial.number}: Check score manually at: https://www.kaggle.com/competitions/{competition}/submissions")
                    
                    # Fallback to CV score if Kaggle score not available
                    cv_scores = []
                    
                    # Setup splitter generator based on strategy
                    if cv_strategy == 'stratified':
                        splitter = cv.split(X_array, y_array)
                    else:
                        splitter = cv.split(X_array, y_array, groups=groups_array)
                        
                    for train_idx, val_idx in splitter:
                        X_fold_train = X_array[train_idx]
                        X_fold_val = X_array[val_idx]
                        y_fold_train = y_array[train_idx]
                        y_fold_val = y_array[val_idx]
                        
                        model_params_cv = params.copy()
                        model_params_cv['early_stopping_rounds'] = 50
                        model_cv = xgb.XGBClassifier(**model_params_cv)
                        model_cv.fit(X_fold_train, y_fold_train,
                                   eval_set=[(X_fold_val, y_fold_val)],
                                   verbose=False)
                        y_pred = model_cv.predict(X_fold_val)
                        acc = accuracy_score(y_fold_val, y_pred)
                        cv_scores.append(acc)
                    score = np.mean(cv_scores)
                    logger.info(f"  Trial {trial.number}: Using CV fallback score: {score:.4f}")
                    logger.info(f"  Trial {trial.number}: Note: Kaggle score may be available later - check website")
                
                # Clean up temp file
                try:
                    os.remove(temp_submission)
                    os.rmdir(temp_dir)
                except:
                    pass
                
                # Final output with score
                final_score = float(score)
                logger.info(f"  Trial {trial.number}: Final objective value: {final_score:.6f}")
                import sys
                sys.stdout.flush()
                
                # Convert score to float and return (Kaggle scores are usually accuracy, maximize)
                return final_score
                
            except Exception as e:
                logger.error(f"  ✗ Trial {trial.number}: Kaggle submission failed: {e}")
                import traceback
                logger.debug(f"  Trial {trial.number}: Traceback: {traceback.format_exc()}")
                # Fallback to CV score
                logger.info(f"  Trial {trial.number}: Falling back to CV score...")
                import sys
                sys.stdout.flush()
                # Continue to CV score mode below
        
        # ========== Use CV Score Mode (default or fallback) ==========
        # Cross-validation
        cv_scores = []
        best_iterations = []
        
        # Setup splitter generator based on strategy
        if cv_strategy == 'stratified':
            splitter = cv.split(X_array, y_array)
        else:
            splitter = cv.split(X_array, y_array, groups=groups_array)
        
        for train_idx, val_idx in splitter:
            X_fold_train = X_array[train_idx]
            X_fold_val = X_array[val_idx]
            y_fold_train = y_array[train_idx]
            y_fold_val = y_array[val_idx]
            
            # Create model with early stopping
            model_params = params.copy()
            model_params['early_stopping_rounds'] = 50
            
            model = xgb.XGBClassifier(**model_params)
            
            # Train with early stopping
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            # Evaluate
            y_pred = model.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(acc)
            
            # Track best iteration
            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                best_iterations.append(model.best_iteration)
            else:
                best_iterations.append(params['n_estimators'])
        
        # Return mean CV accuracy (to maximize)
        mean_acc = np.mean(cv_scores)
        
        # Report intermediate value for pruning
        trial.report(mean_acc, step=len(cv_scores))
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_acc
    
    # Run optimization
    if verbose:
        logger.info("\nStarting optimization...")
        logger.info("  (This may take some time, especially with GPU acceleration)")
        if previous_trials > 0:
            logger.info(f"  Continuing from {previous_trials} previous trials")
            remaining_trials = max(0, n_trials - previous_trials)
            if remaining_trials > 0:
                logger.info(f"  Will run {remaining_trials} additional trials")
            else:
                logger.info(f"  Target of {n_trials} trials already reached")
    
    try:
        # Calculate remaining trials (if continuing from previous study)
        remaining_trials = max(0, n_trials - previous_trials) if previous_trials > 0 else n_trials
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, timeout=timeout, show_progress_bar=True)
        else:
            logger.info(f"  All {n_trials} trials already completed in previous session")
    except KeyboardInterrupt:
        logger.warning("\nOptimization interrupted by user")
        logger.info(f"  Progress saved to: {study_db_path}")
        logger.info(f"  You can resume by running the same command again")
    
    # Extract best parameters
    best_params = study.best_params.copy()
    
    # Add fixed parameters
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 800,  # Will be adjusted based on best_iterations
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
    })
    
    # Add GPU parameters
    best_params.update(base_gpu_params)
    
    # Calculate mean best iterations from final CV run
    # This will be used to set final n_estimators
    # Note: In Kaggle score mode, we still run CV to determine n_estimators for final model
    if verbose:
        logger.info("\nRunning final CV to determine best iterations...")
    final_cv_scores, final_best_iterations = [], []
    
    # Setup splitter generator based on strategy
    if cv_strategy == 'stratified':
        splitter = cv.split(X_array, y_array)
    else:
        splitter = cv.split(X_array, y_array, groups=groups_array)
    
    for train_idx, val_idx in splitter:
        X_fold_train = X_array[train_idx]
        X_fold_val = X_array[val_idx]
        y_fold_train = y_array[train_idx]
        y_fold_val = y_array[val_idx]
        
        model_params_final = best_params.copy()
        model_params_final['early_stopping_rounds'] = 50
        
        model = xgb.XGBClassifier(**model_params_final)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_fold_val)
        acc = accuracy_score(y_fold_val, y_pred)
        final_cv_scores.append(acc)
        
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            final_best_iterations.append(model.best_iteration)
        else:
            final_best_iterations.append(800)
    
    mean_best_iterations = np.mean(final_best_iterations)
    
    # Update configuration with final results
    optimization_config['final_results'] = {
        'best_score': float(study.best_value),
        'best_params': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in study.best_params.items()},
        'total_trials': len(study.trials),
        'mean_best_iterations': float(mean_best_iterations)
    }
    _save_optimization_config(cache_dir, study_id, optimization_config)
    
    if verbose:
        score_type = "Kaggle Score" if use_kaggle_score else "CV Accuracy"
        logger.info(f"\n{'='*60}")
        logger.info("Optimization Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Total trials completed: {len(study.trials)}")
        logger.info(f"Best {score_type}: {study.best_value:.5f} ({study.best_value*100:.2f}%)")
        logger.info(f"Mean Best Iterations: {mean_best_iterations:.1f}")
        logger.info("\nBest Parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\nStudy saved to: {study_db_path}")
        logger.info(f"Config saved to: {cache_dir / f'{study_id}_config.json'}")
        logger.info(f"  You can resume optimization by running the same command again")
        logger.info(f"{'='*60}")
    
    return {
        'best_params': best_params,
        'best_score': study.best_value,
        'study': study,
        'best_iterations': final_best_iterations,
        'mean_best_iterations': mean_best_iterations,
        'study_id': study_id,
        'study_db_path': str(study_db_path),
        'config_path': str(cache_dir / f'{study_id}_config.json'),
        'total_trials': len(study.trials)
    }

