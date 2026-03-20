#!/usr/bin/env bash
# =============================================================================
# Tier 1 A/B Sweep — Single-variable tests against best config (1.1574 bpb)
# Run on RunPod 1×H100. Each experiment ~10 min. Total ~2 hours.
#
# Usage:
#   chmod +x scripts/run_ab_sweep.sh
#   cd parameter-golf/  # must be in the repo with train_gpt.py
#   bash ../scripts/run_ab_sweep.sh [experiment_number]
#
# Pass an experiment number (1-10) to run just that one, or no args for all.
# =============================================================================

set -euo pipefail

LOGDIR="../logs/ab_sweep_$(date +%Y%m%d)"
mkdir -p "$LOGDIR"

# --- Shared base config (current best: MLP3x Int6 QAT zstd sliding window) ---
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export NUM_LAYERS=11
export MLP_MULT=3
export FP16_EMBED_EXPORT=1
export INT6_LAYER_START=0
export INT6_LAYER_END=10
export QAT_ENABLED=1
export QAT_INT6=1
export WARMDOWN_ITERS=3000
export USE_ZSTD=1
export EVAL_STRIDE=64
# These are the "best config" defaults — experiments override one at a time
BASE_MATRIX_LR=0.025
BASE_SCALAR_LR=0.025
BASE_TIED_EMBED_LR=0.035
BASE_MUON_WD=0.04
BASE_ADAM_WD=0.04
BASE_MUON_MOM=0.99
BASE_MUON_MOM_START=0.92
BASE_MUON_MOM_STEPS=1500
BASE_GRAD_CLIP=0.0

run_experiment() {
    local name="$1"
    local logfile="$LOGDIR/${name}.log"

    echo "========================================"
    echo "EXPERIMENT: $name"
    echo "Log: $logfile"
    echo "Started: $(date)"
    echo "========================================"

    torchrun --standalone --nproc_per_node=1 train_gpt.py > "$logfile" 2>&1 || true

    # Extract results
    echo "--- Results for $name ---"
    grep "final_int8_zlib_roundtrip_exact" "$logfile" 2>/dev/null || echo "NO RESULT (crashed?)"
    grep "Total submission size" "$logfile" 2>/dev/null || true
    echo ""
}

reset_to_base() {
    export MATRIX_LR=$BASE_MATRIX_LR
    export SCALAR_LR=$BASE_SCALAR_LR
    export TIED_EMBED_LR=$BASE_TIED_EMBED_LR
    export MUON_WEIGHT_DECAY=$BASE_MUON_WD
    export ADAM_WEIGHT_DECAY=$BASE_ADAM_WD
    export MUON_MOMENTUM=$BASE_MUON_MOM
    export MUON_MOMENTUM_WARMUP_START=$BASE_MUON_MOM_START
    export MUON_MOMENTUM_WARMUP_STEPS=$BASE_MUON_MOM_STEPS
    export GRAD_CLIP_NORM=$BASE_GRAD_CLIP
}

# --- Experiments ---
# Each tests ONE variable change against the best config.
# The best config already has: WD=0.04, Muon=0.99, zstd-22, MLP3x, Int6 QAT.
# So we test variations AROUND the best, plus new features (OrthoInit, EMA, etc).

run_exp_1() {
    # 1. Baseline re-run (control — confirm 1.1574 reproduces on 1×H100)
    reset_to_base
    export RUN_ID="ab01_baseline_rerun"
    export SEED=1337
    run_experiment "01_baseline_rerun"
}

run_exp_2() {
    # 2. Weight decay 0.03 (lower than best's 0.04)
    reset_to_base
    export MUON_WEIGHT_DECAY=0.03
    export ADAM_WEIGHT_DECAY=0.03
    export RUN_ID="ab02_wd_0.03"
    export SEED=1337
    run_experiment "02_wd_0.03"
}

run_exp_3() {
    # 3. Weight decay 0.038 (PR #179 value)
    reset_to_base
    export MUON_WEIGHT_DECAY=0.038
    export ADAM_WEIGHT_DECAY=0.038
    export RUN_ID="ab03_wd_0.038"
    export SEED=1337
    run_experiment "03_wd_0.038"
}

run_exp_4() {
    # 4. Weight decay 0.05 (higher than best's 0.04)
    reset_to_base
    export MUON_WEIGHT_DECAY=0.05
    export ADAM_WEIGHT_DECAY=0.05
    export RUN_ID="ab04_wd_0.05"
    export SEED=1337
    run_experiment "04_wd_0.05"
}

run_exp_5() {
    # 5. Muon momentum 0.95 (revert to baseline default)
    reset_to_base
    export MUON_MOMENTUM=0.95
    export MUON_MOMENTUM_WARMUP_START=0.85
    export MUON_MOMENTUM_WARMUP_STEPS=500
    export RUN_ID="ab05_muon_0.95"
    export SEED=1337
    run_experiment "05_muon_0.95"
}

run_exp_6() {
    # 6. Grad clip 0.3
    reset_to_base
    export GRAD_CLIP_NORM=0.3
    export RUN_ID="ab06_gradclip_0.3"
    export SEED=1337
    run_experiment "06_gradclip_0.3"
}

run_exp_7() {
    # 7. Grad clip 1.0 (milder clip)
    reset_to_base
    export GRAD_CLIP_NORM=1.0
    export RUN_ID="ab07_gradclip_1.0"
    export SEED=1337
    run_experiment "07_gradclip_1.0"
}

run_exp_8() {
    # 8. Higher LR (0.035 matrix, revert toward baseline)
    reset_to_base
    export MATRIX_LR=0.035
    export SCALAR_LR=0.035
    export RUN_ID="ab08_lr_0.035"
    export SEED=1337
    run_experiment "08_lr_0.035"
}

run_exp_9() {
    # 9. Lower LR (0.02 matrix)
    reset_to_base
    export MATRIX_LR=0.02
    export SCALAR_LR=0.02
    export RUN_ID="ab09_lr_0.02"
    export SEED=1337
    run_experiment "09_lr_0.02"
}

run_exp_10() {
    # 10. No zstd (revert to zlib — measure compression penalty)
    reset_to_base
    export USE_ZSTD=0
    export RUN_ID="ab10_zlib_only"
    export SEED=1337
    run_experiment "10_zlib_only"
}

# --- Main ---
if [ $# -ge 1 ]; then
    echo "Running experiment $1 only..."
    run_exp_"$1"
else
    echo "Running all 10 A/B experiments sequentially..."
    echo "Estimated time: ~100 min on 1×H100"
    echo "Logs: $LOGDIR/"
    echo ""
    for i in $(seq 1 10); do
        run_exp_"$i"
    done
fi

# --- Summary ---
echo ""
echo "========================================"
echo "A/B SWEEP COMPLETE — SUMMARY"
echo "========================================"
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    bpb=$(grep "final_int8_zlib_roundtrip_exact" "$logfile" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "FAIL")
    size=$(grep "Total submission size" "$logfile" 2>/dev/null | grep -oP ': \K[0-9]+' || echo "?")
    echo "  $name  bpb=$bpb  size=$size"
done
echo ""
echo "Copy results into docs/DAILY_LOG.md Day 1 table."
