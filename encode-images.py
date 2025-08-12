#!/usr/bin/env python3
# coding: utf-8
"""
Complete feature encoding pipeline for CNN and Transformer models.
Supports:
 - Config-driven model lists or random selection by family
 - Resume logic to skip completed encodings
 - Smart de-duplication of variants vs HuggingFace cache suffixes
 - Local cache loading of weights with safetensors or .bin
 - Extensive logging for transparency

Now accepts:
  --config  Path to YAML config (default: config.yaml)
"""

import os
import yaml
import random
import logging
import math
import shutil
import torch
import timm
import numpy as np
from safetensors.torch import load_file as safe_load
from tqdm import tqdm
from PIL import Image
import difflib
import argparse

# ───────────────────────────────────────────────────────────────
# 0. CLI
# ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Encode images using CNN/Transformer models.")
parser.add_argument("--config", type=str, default="config.yaml",
                    help="Path to the config YAML file (default: config.yaml)")
args = parser.parse_args()

# ───────────────────────────────────────────────────────────────
# 1. Load configuration
# ───────────────────────────────────────────────────────────────
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# ───────────────────────────────────────────────────────────────
# 2. Setup logging
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, cfg.get("logging_level", "INFO").upper()),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# 3. Global paths & settings
# ───────────────────────────────────────────────────────────────
DATASETS_DIR   = cfg["dataset_path"]       # where the datasets reside
ENCODINGS_ROOT = cfg["encodings_dir"]      # where to save .npy encodings & labels
BATCH_SIZE     = cfg.get("batch_size", 64) # images per batch
DEVICE         = torch.device(cfg.get("device", "cpu"))
RESUME         = cfg.get("resume", True)   # skip if final files exist

# log file for failed models.
log_file_path = os.path.join('failed_models.log')
# (optional) clear at start of script:
with open(log_file_path, 'w') as _:
    _.write("Failed Models Log\n=================\n")

# order: 'cnn', 'transformer', or list
order  = cfg["model_type"]
phases = [order.lower()] if isinstance(order, str) else [o.lower() for o in order]

# ───────────────────────────────────────────────────────────────
# 4. Data utility functions
# ───────────────────────────────────────────────────────────────

def get_all_datasets(root):
    """Return list of dataset subdirectories under `root`."""
    return [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]


def get_class_images(ds_path):
    """Walk a dataset folder with class subfolders, return lists of image paths and labels (class indices)."""
    imgs, labs = [], []
    for idx, cls in enumerate(sorted(os.listdir(ds_path))):
        cls_dir = os.path.join(ds_path, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                imgs.append(os.path.join(cls_dir, fn))
                labs.append(idx)
    return imgs, labs

# ───────────────────────────────────────────────────────────────
# 5. Batch encoding function
# ───────────────────────────────────────────────────────────────

def encode_batch(model, transform, images):
    """Apply transform, forward through model, and extract feature vectors."""
    valid_tensors = []
    for p in images:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            logger.warning(f"⚠️ Skipping corrupted image {p}: {e}")
            continue
        try:
            t = transform(img)
        except Exception as e:
            logger.warning(f"⚠️ Transform failed for image {p}: {e}")
            continue
        valid_tensors.append(t)

    if not valid_tensors:
        return np.zeros((0, model.num_features))

    batch = torch.stack(valid_tensors).to(DEVICE)

    with torch.no_grad():
        out = model.forward_features(batch)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    if out.ndim == 3:  # transformer-style (B, N, C)
        feats = out[:, 0]        # CLS token
    elif out.ndim == 4:  # CNN-style (B, C, H, W)
        feats = out.mean(dim=(-2, -1))  # global average pool
    else:
        raise ValueError(f"Unexpected feature shape {out.shape}")

    return feats.cpu().numpy()

# ───────────────────────────────────────────────────────────────
# 6. Optional random model selector by family
# ───────────────────────────────────────────────────────────────

def select_models(num, families):
    """If user did not supply explicit list, pick `num` models evenly from `families`."""
    per = num // len(families)
    rem = num % len(families)
    sel = []

    for fam in families:
        raw = set(timm.list_models(f"*{fam}*", pretrained=True))
        cands = [m for m in raw if m.split('.', 1)[0].startswith(fam)]
        cnt   = per + (1 if rem > 0 else 0)
        rem  -= 1
        if len(cands) <= cnt:
            sel.extend(cands)
        else:
            sel.extend(random.sample(cands, cnt))

    logger.info(f"Selected models: {sel}")
    return sel

# ───────────────────────────────────────────────────────────────
# 7. Main phase runner: handles 'cnn' or 'transformer'
# ───────────────────────────────────────────────────────────────

def run_phase(phase):
    # 7.1 get user-specified list or fallback to random select
    if phase == "transformer":
        models = cfg.get("transformer_models", [])
        n      = cfg.get("num_models", 0)
        fams   = cfg.get("transformer_families", [])
        if not models and n > 0 and fams:
            models = select_models(n, fams)

    elif phase == "cnn":
        models = cfg.get("cnn_models", [])
        models = list(dict.fromkeys(models))
        n      = cfg.get("cnn_num_models", 0)
        fams   = cfg.get("cnn_families", [])
        if not models and n > 0 and fams:
            models = select_models(n, fams)

    else:
        logger.warning(f"Ignoring unknown phase '{phase}'")
        return

    # 7.2 Smart de-duplication
    filtered = []
    seen = set()
    user_list = cfg.get("transformer_models", []) if phase == "transformer" else cfg.get("cnn_models", [])
    for full in models:
        if full in user_list:
            key = full
        else:
            key = full.split('.', 1)[0]
        if key not in seen:
            filtered.append(full)
            seen.add(key)

    models = filtered
    logger.info(f"=== {phase.upper()} phase with models (after de-dup): {models}")
    if not models:
        return

    # 7.3 collect datasets
    datasets = get_all_datasets(DATASETS_DIR)
    if not datasets:
        logger.error(f"No datasets found in {DATASETS_DIR}")
        return

    # 7.4 process each model
    for name in models:
        # check if ALL dataset encodings exist → skip entire model
        if RESUME and all(
            os.path.exists(
                os.path.join(
                    ENCODINGS_ROOT,
                    os.path.basename(ds), phase,
                    name.split('.',1)[0],
                    f"{name.split('.',1)[0]}_encodings.npy"
                )
            ) and os.path.exists(
                os.path.join(
                    ENCODINGS_ROOT,
                    os.path.basename(ds), phase,
                    name.split('.',1)[0],
                    f"{name.split('.',1)[0]}_labels.npy"
                )
            )
            for ds in datasets
        ):
            logger.info(f"Skipping entire model {name} (already done)")
            continue

        # 7.5 load or download model weights (prefer cache)
        model = None
        hf_cache = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub"))
        prefix   = f"models--timm--{name}"

        cached = [d for d in os.listdir(hf_cache)] if os.path.isdir(hf_cache) else []
        cached = [d for d in cached if d.startswith(prefix)]

        try:
            if cached:
                cache_path = os.path.join(hf_cache, cached[0])
                try:
                    wt_file = None
                    for root, _, files in os.walk(cache_path):
                        for f in files:
                            if f.endswith((".safetensors", ".bin")):
                                wt_file = os.path.join(root, f)
                                break
                        if wt_file:
                            break
                    if wt_file is None:
                        raise RuntimeError("No weight file found in cache")
                    logger.info(f"Loading '{name}' from cache at {wt_file}")
                    model = timm.create_model(name, pretrained=False).to(DEVICE).eval()
                    if wt_file.endswith(".safetensors"):
                        sd = safe_load(wt_file, device="cpu")
                    else:
                        sd = torch.load(wt_file, map_location="cpu")
                    model.load_state_dict(sd, strict=False)
                except Exception as e:
                    logger.warning(f"Cache-load failed for '{name}': {e}, will re-download.")
                    shutil.rmtree(cache_path, ignore_errors=True)
                    model = timm.create_model(name, pretrained=True).to(DEVICE).eval()
            else:
                logger.info(f"Downloading pretrained weights for '{name}'")
                model = timm.create_model(name, pretrained=True).to(DEVICE).eval()
        except Exception as e:
            err_str = str(e)
            logger.error(f"❌ Failed to load '{name}' directly: {err_str}")
            all_models = timm.list_models(pretrained=True)
            base = name.split(".", 1)[0]
            primary = [m for m in all_models if m.startswith(base)]
            candidates = primary if primary else difflib.get_close_matches(name, all_models, n=5, cutoff=0.6)
            model = None
            for cand in candidates:
                try:
                    logger.warning(f"Attempting fallback to pretrained '{cand}' for '{name}'")
                    model = timm.create_model(cand, pretrained=True).to(DEVICE).eval()
                    name = cand
                    logger.info(f"✅ Successfully loaded fallback model '{cand}'")
                    break
                except Exception as e2:
                    logger.warning(f"Fallback candidate '{cand}' failed: {e2}")
                    continue
            if model is None:
                logger.error(f"❌ No valid fallback found for '{name}'. Skipping.")
                with open(log_file_path, 'a') as logf:
                    logf.write(f"{name}: No pretrained fallback — {err_str}\n")
                continue

        # 7.6 prepare transforms
        cfg_data  = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**cfg_data)

        # 7.7 encode features for each dataset
        for ds in datasets:
            ds_name = os.path.basename(ds)
            imgs, labs = get_class_images(ds)
            if not imgs:
                logger.warning(f"No images in '{ds_name}', skipping.")
                continue

            base_folder = name.split('.',1)[0]
            model_dir   = os.path.join(ENCODINGS_ROOT, ds_name, phase, base_folder)
            os.makedirs(model_dir, exist_ok=True)

            final_enc = os.path.join(model_dir, f"{base_folder}_encodings.npy")
            final_lbl = os.path.join(model_dir, f"{base_folder}_labels.npy")
            if RESUME and os.path.exists(final_enc) and os.path.exists(final_lbl):
                logger.info(f"Skipping {ds_name}@{base_folder} (done)")
                continue

            total = len(imgs)
            n_batches = math.ceil(total / BATCH_SIZE)
            start = 0
            # resume logic on batch files
            for i in range(n_batches):
                if os.path.exists(os.path.join(model_dir, f"batch_{i}_enc.npy")) and \
                   os.path.exists(os.path.join(model_dir, f"batch_{i}_lbl.npy")):
                    start = i + 1
                else:
                    break

            # process batches
            for i in range(start, n_batches):
                batch_imgs = imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_labs = labs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                logger.info(f"Encoding batch {i+1}/{n_batches} for {ds_name}@{base_folder}")
                feats = encode_batch(model, transform, batch_imgs)
                np.save(os.path.join(model_dir, f"batch_{i}_enc.npy"), feats)
                np.save(os.path.join(model_dir, f"batch_{i}_lbl.npy"), np.array(batch_labs))

            # combine and save final
            all_feats = [
                np.load(os.path.join(model_dir, f"batch_{i}_enc.npy"), allow_pickle=False)
                for i in range(n_batches)
            ]
            all_lbls  = [
                np.load(os.path.join(model_dir, f"batch_{i}_lbl.npy"), allow_pickle=False)
                for i in range(n_batches)
            ]
            encodings = np.concatenate(all_feats, axis=0)
            labels    = np.concatenate(all_lbls, axis=0)

            np.save(final_enc, encodings)
            np.save(final_lbl, labels)
            logger.info(f"Saved final encodings for {ds_name}@{base_folder}")

            # cleanup temporary batch files
            for i in range(n_batches):
                os.remove(os.path.join(model_dir, f"batch_{i}_enc.npy"))
                os.remove(os.path.join(model_dir, f"batch_{i}_lbl.npy"))

# ───────────────────────────────────────────────────────────────
# 8. Entrypoint
# ───────────────────────────────────────────────────────────────

def main():
    random.seed(cfg.get("random_seed", 42))
    for phase in phases:
        run_phase(phase)

if __name__ == "__main__":
    main()
