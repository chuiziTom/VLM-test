#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastparquet",
#     "pandas",
#     "pathlib",
#     "pyarrow",
# ]
# ///
"""
Create parquet files for config subsets of the VSI-Bench dataset.
* debiased: examples kept by the VSI-Bench-Debiased manual filtering pilot (not by automated IBP)
* pruned: examples removed by the VSI-Bench-Debiased manual filtering pilot

> [!NOTE]
> If you do not pass `index=False`, the parquet files will have a `__index_level_0__` column
"""

import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent
pruned_ids_path = script_dir / "pruned_ids.txt"
test_jsonl_path = script_dir / "test.jsonl"
pq_debiased_path = script_dir / "test_debiased.parquet"
pq_pruned_path = script_dir / "test_pruned.parquet"

print("Creating parquet files...")

print(f"Loading pruned ids from '{pruned_ids_path}'...")
with open(pruned_ids_path, "r") as f:
    pruned_ids = f.read().splitlines()
print(f"    -> Loaded {len(pruned_ids)} pruned ids.")

print(f"Loading test data from '{test_jsonl_path}'...")
df = pd.read_json(str(test_jsonl_path), lines=True)
print(f"    -> Loaded {len(df)} examples.")
df["pruned"] = df["id"].astype(str).isin(pruned_ids)
print(f"    -> Added pruned column.")

# save the debiased and pruned subsets separately to parquet files
df_debiased = df[~df["pruned"]]
df_pruned = df[df["pruned"]]

print(f"Saving debiased examples to '{pq_debiased_path}'...")
df_debiased.to_parquet(pq_debiased_path, index=False)
print(f"    -> Saved {len(df_debiased)} debiased examples.")

print(f"Saving pruned examples to '{pq_pruned_path}'...")
df_pruned.to_parquet(pq_pruned_path, index=False)
print(f"    -> Saved {len(df_pruned)} pruned examples.")

print("Done.") 
