#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "$mode" != smoke && "$mode" != train ]]; then
  echo "usage: launch_openwam.sh smoke|train" >&2
  exit 2
fi
work=/mnt/data/benyun/workspace/umi_cup_multimodel_20260925
old=/mnt/data/benyun/workspace/openwam_charger_full_20260922
py="$old/.venv/bin/python"

exec 9>"$work/openwam/training.lock"
flock -n 9 || { echo "another OpenWAM launcher owns the lock" >&2; exit 1; }
"$py" "$work/scripts/gpu_guard.py" --gpu 5
[[ -f "$work/openwam/prepared/manifest.json" ]] || { echo "clean-data preparation missing" >&2; exit 1; }
[[ -f "$work/selection/verified_allowlist.json" ]] || { echo "audited allowlist missing" >&2; exit 1; }

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$work/openwam/torch_extensions_cu128"
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH="$work/scripts:$old/OpenWAM${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled DS_BUILD_OPS=0 MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
cd "$work"

if [[ "$mode" == smoke ]]; then
  [[ ! -e "$work/openwam/smoke" ]] || { echo "smoke directory exists; refusing overwrite" >&2; exit 1; }
  output="$work/openwam/smoke"
  steps=2
  save=2
  full_state=true
else
  [[ -f "$work/openwam/smoke_verified.json" ]] || { echo "smoke checkpoint not verified" >&2; exit 1; }
  [[ ! -e "$work/openwam/checkpoints" ]] || { echo "checkpoint directory exists; refusing overwrite" >&2; exit 1; }
  output="$work/openwam/checkpoints"
  steps=10000
  save=5000
  # The 2-step test proved that full optimizer state can exceed 60 GB and
  # congest the shared NFS. Formal milestones only need deployable weights.
  full_state=false
fi

exec "$py" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$work/scripts/train_openwam_cup.py" \
  '+model.freeze=[video_backbone]' dataloader=charger \
  dataloader.type=umi_cup_clean \
  dataloader.dataset_dir="$work/openwam/prepared" \
  dataloader.prepared_dir="$work/openwam/prepared" \
  'dataloader.unify_action_map=["0-9","34-43"]' \
  training.finetune_ckpt_path="$old/foundation" training.resume_ckpt_path=null \
  training.batch_size=1 training.gradient_accumulation_steps=1 \
  training.num_epochs=null training.max_steps="$steps" \
  training.mixed_precision=bf16 training.zero_stage=2 \
  training.offload_optimizer_device=cpu training.initialize_model_on_cpu=true \
  training.use_gradient_checkpointing=true training.use_gradient_checkpointing_offload=true \
  training.learning_rate=1e-5 training.lr_scheduler=null \
  training.dataset_num_workers=0 training.save_steps="$save" \
  training.save_full_states_for_resume="$full_state" training.keep_last_k_ckpts=3 \
  training.output_path="$output" project.output_dir="$output" \
  project.wandb.run_name="umi_cup_clean_openwam_${mode}"
