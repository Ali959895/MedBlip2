#!/usr/bin/env bash
#SBATCH -J Kvasir_mix
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --constraint=gpu48
#SBATCH --time=7-00:00:00
#SBATCH --output=/nfs/speed-scratch/%u/logs/Kvasir-%j.out
#SBATCH --error=/nfs/speed-scratch/%u/logs/Kvasir-%j.err
#SBATCH --chdir=/nfs/speed-scratch/%u/kvasir_finetuning_v3

set -euo pipefail
# -------------------------
# Scratch paths
# -------------------------
USER_SCR=/nfs/speed-scratch/${USER}
PROJ=${USER_SCR}/kvasir_finetuning_v3

CFG_BLIP2=${PROJ}/configs/Kvasir_finetune.yaml
CFG_BLIP=${PROJ}/configs/Kvasir_blip.yaml

# LAVIS inside HAM10000 repo (as you have it)
LAVIS_DIR=${USER_SCR}/HAM10000_finetuning/LAVIS

# Envs (these are conda env folders, but we will NOT "conda activate")
BLIP2_ENV=${USER_SCR}/blip2int-env
BLIP_ENV=${USER_SCR}/venvs/blip-env

mkdir -p "${USER_SCR}/logs" "${USER_SCR}/wandb" "${USER_SCR}/hf_home" "${USER_SCR}/cache/torch"

# -------------------------
# Caches / offline / wandb
# -------------------------
export WANDB_DIR=${USER_SCR}/wandb
export HF_HOME=${USER_SCR}/hf_home
export TRANSFORMERS_CACHE=${HF_HOME}
export TORCH_HOME=${USER_SCR}/cache/torch
export TOKENIZERS_PARALLELISM=false

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
export WANDB_SILENT=true
export WANDB_INIT_TIMEOUT=300

# -------------------------
# PYTHONPATH (bash syntax only)
# -------------------------
export PYTHONPATH="${LAVIS_DIR}:${PROJ}/src:${PYTHONPATH:-}"

echo "HOST=$(hostname)"
echo "PROJ=${PROJ}"
echo "LAVIS_DIR=${LAVIS_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"

# -------------------------


# Helper: run command using env's python without conda/module
# -------------------------
run_with_env () {
  local ENV_PREFIX="$1"; shift
  if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "[ERROR] Missing python: ${ENV_PREFIX}/bin/python"
    echo "Contents of ${ENV_PREFIX}:"
    ls -lah "${ENV_PREFIX}" || true
    exit 2
  fi

  # Prepend env bin so it uses correct python/pip packages
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  hash -r

  echo "=== Using python: $(which python) ==="
  python -V
  "$@"
}


# Run 2/2: BLIP env
# -------------------------
echo "=== Run 2/2: BLIP env ==="
if [[ -x "${BLIP_ENV}/bin/python" ]]; then
  run_with_env "${BLIP_ENV}" python -c "import lavis; print('lavis ok (blip)')"
  run_with_env "${BLIP_ENV}" python "${PROJ}/src/run.py" -c "${CFG_BLIP}" --mode benchmark_multiclass
else
  echo "[WARN] BLIP env is broken (missing ${BLIP_ENV}/bin/python). Skipping Run 2/2."
  ls -lah "${BLIP_ENV}" || true
fi

# -------------------------
# Run 1/2: BLIP2 env
# -------------------------
echo "=== Run 1/2: BLIP2 env ==="
run_with_env "${BLIP2_ENV}" python -c "import lavis; print('lavis ok (blip2)')"
run_with_env "${BLIP2_ENV}" python "${PROJ}/src/run.py" -c "${CFG_BLIP2}" --mode benchmark_multiclass
#run_with_env "${BLIP2_ENV}" python "${PROJ}/src/run.py" -c "${CFG_BLIP2}" --mode crossval_multiclass
#python src/run.py -c configs/Kvasir_finetune.yaml --mode crossval_multiclass

# -------------------------
# -------------------------


#echo "=== Run 2/2: BLIP env ==="
#run_with_env "${BLIP_ENV}" python -c "import lavis; print('lavis ok (blip)')"
#run_with_env "${BLIP_ENV}" python "${PROJ}/src/run.py" -c "${CFG_BLIP}" --mode benchmark_multiclass

echo "=== DONE ==="
