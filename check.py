"""
Quick script to check user overlap between train.jsonl and kaggle_test.jsonl.

This version uses the existing data_processing.load_data utility, which already
handles the JSONL structure and flattening logic correctly.
"""

from src.utils import data_processing as dp  # type: ignore

# Absolute paths in this environment
train_path = "/Data/yuguang.yao/-quipe-LLM-projet-de-DL-2025/data/train.jsonl"
test_path = "/Data/yuguang.yao/-quipe-LLM-projet-de-DL-2025/data/kaggle_test.jsonl"


def collect_user_ids_from_df(path: str) -> set:
    """Collect unique user ids from a JSONL file via flattened DataFrame."""
    df = dp.load_data(path, clean_text=True)

    user_col = None
    # Prefer numeric user.id, fall back to user.id_str if needed
    if "user.id" in df.columns:
        user_col = "user.id"
    elif "user.id_str" in df.columns:
        user_col = "user.id_str"
    else:
        # Last‑resort heuristic: any column that ends with these patterns
        candidates = [
            c
            for c in df.columns
            if c.endswith("user.id") or c.endswith("user.id_str")
        ]
        if candidates:
            user_col = candidates[0]

    if user_col is None:
        print(f"No user id column found in {path}")
        return set()

    user_series = df[user_col].dropna().astype(str)
    return set(user_series.tolist())


if __name__ == "__main__":
    train_users = collect_user_ids_from_df(train_path)
    test_users = collect_user_ids_from_df(test_path)

    overlap = train_users & test_users

    print(f"Train unique users: {len(train_users)}")
    print(f"Test unique users:  {len(test_users)}")
    print(f"Overlap users:       {len(overlap)}")
    if train_users:
        print(f"Overlap / train: {len(overlap) / len(train_users):.4f}")
    if test_users:
        print(f"Overlap / test:  {len(overlap) / len(test_users):.4f}")