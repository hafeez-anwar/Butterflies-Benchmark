#!/usr/bin/env python3
"""
Main pipeline runner for:
  1) End-to-end: Encoding + Classification
  2) Encoding only
  3) Classification only

Both child scripts accept:  --config <path>   (default: config.yaml)

Examples:
  python main.py --mode 1
  python main.py --mode all --config config.yaml
  python main.py --mode encode
  python main.py --mode classify --config my-experiment.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENCODER = HERE / "encode-images.py"
CLASSIFIER = HERE / "classify.py"

def run_script(script_path: Path, config_path: str) -> None:
    if not script_path.exists():
        sys.exit(f"❌ Missing script: {script_path}")
    print(f"\n=== Running {script_path.name} with config: {config_path} ===")
    result = subprocess.run(
        [sys.executable, str(script_path), "--config", config_path],
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=str(HERE)
    )
    if result.returncode != 0:
        sys.exit(f"❌ {script_path.name} failed with exit code {result.returncode}")
    print(f"✅ Finished {script_path.name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Driver for encoding/classification pipeline.")
    parser.add_argument(
        "--mode",
        "-m",
        default="all",
        help="1|2|3 or all|encode|classify (default: all). "
             "1=all, 2=encode, 3=classify"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    return parser.parse_args()

def normalize_mode(raw: str) -> str:
    m = str(raw).strip().lower()
    if m in {"1", "all", "endtoend", "end-to-end", "e2e"}:
        return "all"
    if m in {"2", "encode", "encoding"}:
        return "encode"
    if m in {"3", "classify", "classification"}:
        return "classify"
    sys.exit("❌ Unknown --mode. Use 1|2|3 or all|encode|classify.")

def main():
    args = parse_args()
    mode = normalize_mode(args.mode)
    config_path = args.config

    # Print selected option
    label = {"all": "1 - End to End encoding + classification",
             "encode": "2 - Just encoding",
             "classify": "3 - Just classification"}[mode]
    print(f"Selected: {label}")
    print(f"Config:   {config_path}")

    if mode in {"all", "encode"}:
        run_script(ENCODER, config_path)

    if mode in {"all", "classify"}:
        run_script(CLASSIFIER, config_path)

    print("\n🎉 Done.")

if __name__ == "__main__":
    main()
