#!/usr/bin/env bash
# Runs the entire FinCausalSAE pipeline in DEMO mode (CPU, gpt2, synthetic
# data) end-to-end. Takes a few minutes on a normal laptop. Use this to
# confirm your environment is set up correctly before ever touching a GPU.
#
# Usage:
#   bash run_demo.sh
set -euo pipefail

echo "=================================================================="
echo " FinCausalSAE — full demo pipeline (CPU, ~5-15 minutes)"
echo "=================================================================="

python3 data/00_collect_data.py --mode demo
python3 models/01_lora_finetune.py --mode demo
python3 sae/02_train_sae.py --mode demo
python3 circuits/03_causal_patching.py --mode demo
python3 backtest/04_portfolio_backtest.py --mode demo

echo "=================================================================="
echo " Demo pipeline finished. Check:"
echo "   circuits/results/causal_scores.parquet"
echo "   circuits/results/causal_feature_library.json"
echo "   backtest/results/summary.json"
echo "   backtest/results/portfolio_comparison.png"
echo "=================================================================="
