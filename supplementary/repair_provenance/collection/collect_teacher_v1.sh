#!/usr/bin/env bash
set -uo pipefail
cd ~/SimplerEnv_stable
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate simpler_octo_stable
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH || true
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 JAX_THREEFRY_PARTITIONABLE=false TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=1 MPLBACKEND=Agg
export OCTO_WIDOWX_GRIPPER_MODE=threshold_075
export OCTO_SAVE_VIDEO=0
BASE=./rollout_teacher_collect_v1
mkdir -p "$BASE"
EIDS="0 1 2 9 10 11 12 13 14 21 22 23"
echo "COLLECT_START $(date +%s) base=$BASE eids=[$EIDS] seeds=100..179"
for seed in $(seq 100 179); do
  export OCTO_ROLLOUT_RECORD_DIR="$BASE/seed_$seed"
  mkdir -p "$BASE/seed_$seed/results"
  if python simpler_env/main_inference.py \
      --policy-model octo-base --ckpt-path hf://rail-berkeley/octo-base-1.5 \
      --robot widowx_sink_camera_setup --policy-setup widowx_bridge \
      --octo-init-rng "$seed" --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutEggplantInBasketScene-v0 --scene-name bridge_table_1_v2 \
      --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
      --robot-init-x 0.127 0.127 1 --robot-init-y 0.06 0.06 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --obj-variation-mode episode --obj-episode-list $EIDS \
      --logging-dir "$BASE/seed_$seed/results" --additional-env-save-tags "collect_seed$seed" \
      > "$BASE/seed_$seed.log" 2>&1; then
    echo "SEED $seed DONE $(date +%s)"
  else
    echo "SEED $seed FAILED $(date +%s)"
  fi
done
echo "COLLECT_ALL_DONE $(date +%s)"
