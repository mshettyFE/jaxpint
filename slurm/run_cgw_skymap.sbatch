#!/bin/bash
#SBATCH --account=<TODO_FILL_IN>
#SBATCH --job-name=jaxpint-cgw-skymap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# CGW distance lower-limit sky map (no-MCMC Bayesian approx of Fig. 8 of
# arXiv:2306.16222). Runs examples/cgw_distance_skymap.py inside the same
# Apptainer overlay as run_distance_scan.sbatch — no new Python deps.
#
# Submit from the JaxPINT repo root with:
#     sbatch slurm/run_cgw_skymap.sbatch
#
# Before first submission:
#   1. Replace <TODO_FILL_IN> above with your account (`my_slurm_accounts`).
#   2. Build the overlay once (shared with the distance scan):
#        srun --cpus-per-task=2 --mem=10G --time=2:00:00 --pty bash
#        bash slurm/build_overlay.sh
#   3. Stage the ocarina dataset onto /scratch. From your LAPTOP:
#        bash slurm/stage_ocarina.sh <netid>
#      (ocarina is the small synthetic par/tim set; ~63 MB.)

set -euo pipefail

echo "[sbatch] Host: $(hostname)"
echo "[sbatch] Job:  ${SLURM_JOB_ID}"
printenv | grep -E '^SLURM_' | sort

module purge

# Don't preallocate the GPU (matches local behavior; avoids autotuner OOM).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Which staged dataset to run, and in which likelihood mode. Override at submit:
#   sbatch --export=ALL,OCARINA=ocarina,DATA_MODE=real slurm/run_cgw_skymap.sbatch
OCARINA="${OCARINA:-ocarina_2}"
DATA_MODE="${DATA_MODE:-real}"

# Ocarina par/tim directory staged by slurm/stage_ocarina.sh (mirrors the local
# dataset name). The driver reads this via $JAXPINT_OCARINA_DIR.
export JAXPINT_OCARINA_DIR="${JAXPINT_OCARINA_DIR:-/scratch/${USER}/jaxpint-data/${OCARINA}}"

OVERLAY="/scratch/${USER}/jaxpint/overlay-15GB-500K.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

OUT_DIR="/scratch/${USER}/jaxpint-out"
OUT_FILE="${OUT_DIR}/cgw-skymap-${OCARINA}-${DATA_MODE}-${SLURM_JOB_ID}.npz"
mkdir -p "${OUT_DIR}"

# Resolution / fidelity knobs (override by editing these).
NSIDE="${NSIDE:-8}"          # HEALPix nside (npix = 12*nside^2; 8 -> 768 pixels)

# Sanity-check the artefacts we depend on; surface clear errors early.
for path in "${OVERLAY}" "${IMAGE}" "${PROJECT_DIR}/examples/cgw_distance_skymap.py" "${JAXPINT_OCARINA_DIR}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[sbatch] ERROR: required path missing: ${path}" >&2
        exit 1
    fi
done

echo "[sbatch] OVERLAY     = ${OVERLAY}"
echo "[sbatch] IMAGE       = ${IMAGE}"
echo "[sbatch] PROJECT_DIR = ${PROJECT_DIR}"
echo "[sbatch] OCARINA_DIR = ${JAXPINT_OCARINA_DIR}"
echo "[sbatch] DATA_MODE   = ${DATA_MODE}"
echo "[sbatch] OUT_FILE    = ${OUT_FILE}"
echo "[sbatch] NSIDE       = ${NSIDE}"

nvidia-smi || true

# `generate` only — pure compute, no matplotlib. Plot from the .npz afterwards.
# `--full` uses all pulsars (B1937+21 telescope variants dropped automatically).
srun singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${OUT_DIR}:${OUT_DIR}" \
    --bind "${JAXPINT_OCARINA_DIR}:${JAXPINT_OCARINA_DIR}" \
    --env "JAXPINT_OCARINA_DIR=${JAXPINT_OCARINA_DIR}" \
    --env "XLA_PYTHON_CLIENT_PREALLOCATE=false" \
    --env "PYTHONHTTPSVERIFY=0"\ # Hack to get clock ata
    "${IMAGE}" \
    /bin/bash -c "source /ext3/env.sh && cd '${PROJECT_DIR}' && python examples/cgw_distance_skymap.py generate --output '${OUT_FILE}' --nside '${NSIDE}' --full --data-mode '${DATA_MODE}'"

echo "[sbatch] Done. Output at ${OUT_FILE}"
