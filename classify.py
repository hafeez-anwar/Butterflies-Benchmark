#!/usr/bin/env python3
# coding: utf-8
"""
Classification with (Repeated) Stratified K-Fold on top of precomputed encodings.

Reads encodings produced by encode-images.py in the structure:
  encodings_dir/
    <dataset_name>/
      cnn/ or transformer/
        <model_base>/
          <model_base>_encodings.npy
          <model_base>_labels.npy

Outputs per-model metrics CSVs and an optional Excel summary.
If save_final_models: saves {dataset}_{phase}_{model}.joblib and _classes.npy.

Now accepts:
  --config  Path to YAML config (default: config.yaml)
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import argparse

# ------------------------------
# CLI & Config
# ------------------------------
parser = argparse.ArgumentParser(description="SVM classification over precomputed encodings.")
parser.add_argument("--config", type=str, default="config.yaml",
                    help="Path to the config YAML file (default: config.yaml)")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# ------------------------------
# Helpers
# ------------------------------

def list_datasets(enc_root: Path):
    return [p for p in enc_root.iterdir() if p.is_dir()]

def list_models_for_phase(ds_dir: Path, phase: str):
    phase_dir = ds_dir / phase
    if not phase_dir.exists():
        return []
    return [p for p in phase_dir.iterdir() if p.is_dir()]

def load_xy(model_dir: Path):
    base = model_dir.name.split('.', 1)[0]
    X = np.load(model_dir / f"{base}_encodings.npy")
    y = np.load(model_dir / f"{base}_labels.npy")
    return X, y

def topk_accuracy(prob: np.ndarray, y_true: np.ndarray, k: int) -> float:
    # prob shape: (N, C)
    k = min(k, prob.shape[1])
    topk = np.argpartition(-prob, kth=k-1, axis=1)[:, :k]
    hits = np.any(topk == y_true[:, None], axis=1)
    return float(hits.mean())

# ------------------------------
# Main
# ------------------------------

def main():
    enc_root = Path(cfg["encodings_dir"]).resolve()
    results_root = Path(cfg.get("results_dir", "results")).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    n_splits = int(cfg.get("kfold", {}).get("n_splits", 5))
    n_repeats = int(cfg.get("kfold", {}).get("n_repeats", 1))
    random_state = int(cfg.get("random_state", 42))
    save_models = bool(cfg.get("save_final_models", True))

    phases = cfg.get("model_type")
    phases = [phases] if isinstance(phases, str) else phases

    all_rows = []

    for ds_dir in list_datasets(enc_root):
        ds_name = ds_dir.name
        for phase in phases:
            model_dirs = list_models_for_phase(ds_dir, phase)
            for model_dir in model_dirs:
                model_name = model_dir.name
                base = model_name.split('.', 1)[0]
                print(f"\n=== Dataset: {ds_name} | Phase: {phase} | Model: {model_name} ===")
                X, y = load_xy(model_dir)
                n_classes = len(np.unique(y))

                # CV splitter
                if n_repeats > 1:
                    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                else:
                    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

                # Pipeline: StandardScaler + Linear SVM (with probability)
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="linear", probability=True, random_state=random_state)),
                ])

                fold_rows = []
                fold = 0
                for tr_idx, te_idx in splitter.split(X, y):
                    fold += 1
                    Xtr, Xte = X[tr_idx], X[te_idx]
                    ytr, yte = y[tr_idx], y[te_idx]
                    pipe.fit(Xtr, ytr)
                    prob = pipe.predict_proba(Xte)
                    ypred = np.argmax(prob, axis=1)

                    acc = accuracy_score(yte, ypred)
                    prec, rec, f1, _ = precision_recall_fscore_support(yte, ypred, average='macro', zero_division=0)
                    top1 = topk_accuracy(prob, yte, k=1)
                    top3 = topk_accuracy(prob, yte, k=3) if n_classes >= 3 else np.nan
                    top5 = topk_accuracy(prob, yte, k=5) if n_classes >= 5 else np.nan

                    fold_rows.append({
                        "dataset": ds_name,
                        "phase": phase,
                        "model": model_name,
                        "fold": fold,
                        "acc": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "top1": top1,
                        "top3": top3,
                        "top5": top5,
                    })

                # Save per-model CSV
                df = pd.DataFrame(fold_rows)
                out_dir = results_root / ds_name / phase
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir / f"{base}_kfold_metrics.csv", index=False)

                # Aggregate row
                agg = df.groupby(["dataset", "phase", "model"]).mean(numeric_only=True).reset_index()
                mean_row = agg.iloc[0].to_dict()
                mean_row.update({"folds": len(df)})
                all_rows.append(mean_row)

                # Optionally train final model on ALL data and save
                if save_models:
                    pipe.fit(X, y)
                    joblib.dump(pipe, out_dir / f"{base}_svm.joblib")
                    np.save(out_dir / f"{base}_classes.npy", np.unique(y))

    # Save overall summary
    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary.to_csv(results_root / "summary_kfold_all_models.csv", index=False)
        try:
            summary.to_excel(results_root / "svm_classification_results.xlsx", index=False)
        except Exception:
            pass

    print("\nDone: K-Fold classification finished.")

if __name__ == "__main__":
    main()
