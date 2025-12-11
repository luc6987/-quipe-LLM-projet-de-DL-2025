#!/usr/bin/python3
"""
Text Classification CV Runner
文本五折交叉验证 + Optuna 超参脚本

Generates reliable CV metrics for the DeBERTa text model and optionally runs a
lightweight Optuna sweep before full cross-validation.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import data_processing as dp
from src.utils.bertweet_lora import train_lora as tl
from src.utils.paths import PROJECT_ROOT, DATA_DIR, get_models_output_dir


def load_text_and_labels(train_path: str):
    df = dp.load_data(train_path, clean_text=True)
    if 'label' not in df.columns:
        raise ValueError("Training data must include 'label' column.")
    if 'text' not in df.columns:
        if 'extended_tweet.full_text' in df.columns:
            df['text'] = df['extended_tweet.full_text'].fillna('')
        else:
            raise ValueError("No text column found (expected 'text' or 'extended_tweet.full_text').")
    texts = df['text'].fillna('').astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels


def run_optuna_search(train_dataset, val_dataset, tokenizer, data_collator, args):
    if not tl.OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed. Please install optuna to enable --n_trials.")
    import optuna

    def _objective(trial):
        return tl.objective(
            trial,
            train_dataset,
            val_dataset,
            tokenizer,
            data_collator,
            model_name=args.model_name,
            use_lora=args.use_lora,
            target_modules=args.lora_target_modules,
            focal_gamma=args.focal_gamma,
            rdrop_alpha=args.rdrop_alpha,
            gradient_checkpointing=args.gradient_checkpointing
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=args.n_trials)
    best_params = study.best_trial.params
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "best_optuna_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    return best_params


def parse_args():
    parser = argparse.ArgumentParser(description="DeBERTa Text CV Runner")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--output_dir", type=str, default=str(get_models_output_dir() / "text_cv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=0, help="Run Optuna trials before CV (0 to skip)")
    parser.add_argument("--params_file", type=str, default=None, help="JSON file with tuning parameters")
    parser.add_argument("--model_name", type=str, default=tl.MODEL_NAME)
    parser.add_argument("--use_lora", type=tl.str2bool, default=tl.USE_LORA)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--rdrop_alpha", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=tl.MAX_LENGTH)
    parser.add_argument("--learning_rate", type=float, default=tl.LEARNING_RATE)
    parser.add_argument("--lora_r", type=int, default=tl.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=tl.LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=tl.LORA_DROPOUT)
    parser.add_argument("--batch_size", type=int, default=tl.BATCH_SIZE)
    parser.add_argument("--weight_decay", type=float, default=tl.WEIGHT_DECAY)
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = os.path.join(args.data_dir, args.train_file)
    os.makedirs(args.output_dir, exist_ok=True)

    texts, labels = load_text_and_labels(train_path)
    tokenizer = tl.get_tokenizer(args.model_name)
    dataset = tl.create_dataset_from_text(texts, labels, tokenizer=tokenizer, max_length=args.max_length)
    data_collator = tl.DataCollatorWithPadding(tokenizer=tokenizer)

    params = {
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay
    }

    if args.params_file and os.path.exists(args.params_file):
        with open(args.params_file, "r") as f:
            params.update(json.load(f))
        print(f"Loaded parameters from {args.params_file}")
    elif args.n_trials > 0:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=tl.SEED)
        first_train_idx, first_val_idx = next(skf.split(np.zeros(len(labels)), labels))
        opt_train = dataset.select(first_train_idx.tolist())
        opt_val = dataset.select(first_val_idx.tolist())
        best_params = run_optuna_search(opt_train, opt_val, tokenizer, data_collator, args)
        params.update(best_params)

    print("Using parameters:", params)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=tl.SEED)
    fold_accuracies = []
    oof_logits = np.zeros(len(labels), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        train_dataset = dataset.select(train_idx.tolist())
        val_dataset = dataset.select(val_idx.tolist())

        trainer, val_acc = tl.train_model_with_params(
            learning_rate=params["learning_rate"],
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            batch_size=params["batch_size"],
            weight_decay=params["weight_decay"],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            output_dir=fold_dir,
            model_name=args.model_name,
            use_lora=args.use_lora,
            target_modules=args.lora_target_modules,
            focal_gamma=args.focal_gamma,
            rdrop_alpha=args.rdrop_alpha,
            gradient_checkpointing=args.gradient_checkpointing
        )

        trainer.save_model(fold_dir)
        eval_metrics = trainer.evaluate()
        fold_accuracy = eval_metrics.get("eval_accuracy", val_acc)
        fold_accuracies.append(fold_accuracy)

        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions
        if logits.ndim == 2 and logits.shape[1] >= 2:
            fold_logits = logits[:, 1] - logits[:, 0]
        else:
            fold_logits = logits.squeeze()
        oof_logits[val_idx] = fold_logits

        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump({"accuracy": fold_accuracy}, f, indent=4)

        torch.cuda.empty_cache()

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    summary = {
        "model_name": args.model_name,
        "use_lora": args.use_lora,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "params": params
    }
    with open(os.path.join(args.output_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    np.save(os.path.join(args.output_dir, "oof_logits.npy"), oof_logits)

    log_path = os.path.join(project_root, ".cursor", "memory", "experiment_log.md")
    log_line = f"| 2025-11-26 | TEXT-CV-DeBERTa | {mean_acc:.4f}±{std_acc:.4f} | N/A | {args.model_name} {args.folds}-fold CV |\n"
    with open(log_path, "a") as log_file:
        log_file.write(log_line)


if __name__ == "__main__":
    main()

