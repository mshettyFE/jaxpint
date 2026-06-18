#!/bin/bash
#SBATCH --account=<TODO_FILL_IN>
#SBATCH --job-name=jaxpint-cgw-multi-source
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Phase 2: K=2 multi-source CGW localization. Source 0 scanned over HEALPix;
# source 1 held at a canonical galaxy-cluster direction (default: Coma).
# Per-source 90% / 50% credible areas across the three Wen array configs.
#
# Two modes via MODE env var:
#   MODE=smoke      (default if K=2 + small nside) — single-config end-to-end run.
#                   Used to validate the pipeline before committing the full sweep.
#                   Defaults to 2-pulsar SMOKE_SUBSET, nside=2, no anchors.
#   MODE=sweep      — primary Phase 2 deliverable. Three Wen configs
#                   (standard / 25-3 / 25-6) on WEN_OCARINA_18, nside=4.
#                   Outputs the Wen-Table-1-multi-source-analog CSV.
#
# Memory is HIGHER than Level 1 (256G -> 512G) because the joint Fisher's
# K(K+1)/2 = 3 cross-derivative blocks per pixel scale the autodiff tape
# roughly as K². PIXEL_CHUNK is correspondingly smaller (16 -> 8).
#
# Submission examples (run from the JaxPINT repo root on cluster):
#   # Smoke first (quick sanity check, ~30 min)
#   sbatch --export=ALL,MODE=smoke slurm/run_cgw_multi_source.sbatch
#
#   # Full Phase 2 sweep (~1-3 h)
#   sbatch --export=ALL,MODE=sweep slurm/run_cgw_multi_source.sbatch
#
#   # Smoke → sweep with dependency (sweep auto-runs only if smoke succeeds)
#   SMOKE_ID=$(sbatch --parsable --export=ALL,MODE=smoke slurm/run_cgw_multi_source.sbatch)
#   sbatch --dependency=afterok:$SMOKE_ID --export=ALL,MODE=sweep slurm/run_cgw_multi_source.sbatch

set -euo pipefail

echo "[sbatch] Host: $(hostname)"
echo "[sbatch] Job:  ${SLURM_JOB_ID}"
printenv | grep -E '^SLURM_' | sort

module purge

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Mode and knobs.
MODE="${MODE:-smoke}"
OCARINA="${OCARINA:-ocarina}"
K="${K:-2}"
FIXED_SOURCE_NAMES="${FIXED_SOURCE_NAMES:-Coma}"  # space-separated for K > 2

if [ "${MODE}" = "smoke" ]; then
    NSIDE="${NSIDE:-2}"
    PIXEL_CHUNK="${PIXEL_CHUNK:-4}"
elif [ "${MODE}" = "sweep" ]; then
    NSIDE="${NSIDE:-4}"
    PIXEL_CHUNK="${PIXEL_CHUNK:-8}"
else
    echo "[sbatch] ERROR: unknown MODE='${MODE}' (expected 'smoke' or 'sweep')" >&2
    exit 1
fi
SNR_TARGET="${SNR_TARGET:-20.0}"

export JAXPINT_OCARINA_DIR="${JAXPINT_OCARINA_DIR:-/scratch/${USER}/jaxpint-data/${OCARINA}}"

OVERLAY="/scratch/${USER}/jaxpint/overlay-15GB-500K.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

OUT_DIR="/scratch/${USER}/jaxpint-out/cgw-multi-${OCARINA}-${MODE}-K${K}-${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

for path in "${OVERLAY}" "${IMAGE}" \
            "${PROJECT_DIR}/examples/cgw_multi_source_localization.py" \
            "${JAXPINT_OCARINA_DIR}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[sbatch] ERROR: required path missing: ${path}" >&2
        exit 1
    fi
done

echo "[sbatch] OVERLAY            = ${OVERLAY}"
echo "[sbatch] IMAGE              = ${IMAGE}"
echo "[sbatch] PROJECT_DIR        = ${PROJECT_DIR}"
echo "[sbatch] OCARINA_DIR        = ${JAXPINT_OCARINA_DIR}"
echo "[sbatch] MODE               = ${MODE}"
echo "[sbatch] K                  = ${K}"
echo "[sbatch] FIXED_SOURCE_NAMES = ${FIXED_SOURCE_NAMES}"
echo "[sbatch] NSIDE              = ${NSIDE}"
echo "[sbatch] PIXEL_CHUNK        = ${PIXEL_CHUNK}"
echo "[sbatch] SNR_TARGET         = ${SNR_TARGET}"
echo "[sbatch] OUT_DIR            = ${OUT_DIR}"

nvidia-smi || true

if [ "${MODE}" = "smoke" ]; then
    # Single-config 2-pulsar smoke. Saves one .npz; quick end-to-end pipeline check.
    OUT_FILE="${OUT_DIR}/cgw_loc_multi_K${K}_smoke.npz"
    CLI="python examples/cgw_multi_source_localization.py generate \
        --output '${OUT_FILE}' \
        --K '${K}' \
        --fixed-source-names ${FIXED_SOURCE_NAMES} \
        --nside '${NSIDE}' \
        --snr '${SNR_TARGET}' \
        --pixel-chunk '${PIXEL_CHUNK}'"
else
    # Sweep: three Wen configs on WEN_OCARINA_18, primary Phase 2 deliverable.
    CLI="python examples/cgw_multi_source_localization.py sweep \
        --out-dir '${OUT_DIR}' \
        --K '${K}' \
        --fixed-source-names ${FIXED_SOURCE_NAMES} \
        --nside '${NSIDE}' \
        --snr '${SNR_TARGET}' \
        --pixel-chunk '${PIXEL_CHUNK}'"
fi

srun singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${OUT_DIR}:${OUT_DIR}" \
    --bind "${JAXPINT_OCARINA_DIR}:${JAXPINT_OCARINA_DIR}" \
    --env "JAXPINT_OCARINA_DIR=${JAXPINT_OCARINA_DIR}" \
    --env "XLA_PYTHON_CLIENT_PREALLOCATE=false" \
    --env "PYTHONHTTPSVERIFY=0" \
    --env "PYTHONPATH=${PROJECT_DIR}" \
    "${IMAGE}" \
    /bin/bash -c "source /ext3/env.sh && cd '${PROJECT_DIR}' && ${CLI}"

echo "[sbatch] Done. Outputs in ${OUT_DIR}"
