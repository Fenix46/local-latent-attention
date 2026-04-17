#!/usr/bin/env bash
# Torchrun launcher per training DDP single-node.
#
# Uso:
#   ./launch_ddp.sh --config ... --preset 0.55b ...
#
# Variabili d'ambiente:
#   NPROC          numero di GPU sul nodo (default: tutte quelle visibili)
#   MASTER_PORT    porta rendezvous (default: 29500)
#   NCCL_DEBUG     INFO per diagnosi collettive (default: WARN)
#
# Fallback single-process: se NPROC=1 torchrun gira comunque, e distributed.py
# riconosce launched_under_torchrun() quindi il path DDP si attiva con
# world_size=1 — utile per lo smoke test di parità.

set -euo pipefail

# Conta GPU visibili se NPROC non impostato.
if [[ -z "${NPROC:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
  else
    NPROC=1
  fi
fi

export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Riduce frammentazione dell'allocatore CUDA sui long-run.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[launch_ddp] nproc_per_node=${NPROC} master_port=${MASTER_PORT}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train.py "$@"
