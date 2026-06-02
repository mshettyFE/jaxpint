#!/bin/bash
#SBATCH --account=<TODO_FILL_IN>
#SBATCH --job-name=jaxpint-cgw-localization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# CGW sky-localization map vs. anchor-pulsar count (Wen et al. 2026 style;
# arXiv:2603.28897). Runs examples/cgw_localization_skymap.py inside the same
# Apptainer overlay as run_cgw_skymap.sbatch (the items-1-5 distance map) —
# no new Python deps.
#
# Memory is higher than the distance-map sbatch because the localization Fisher
# uses a Hessian of Y(sky) (a 4th-order autodiff tape through marginalize), so
# the per-chunk grad tape is much larger. PIXEL_CHUNK is correspondingly smaller.
#
# Submit from the JaxPINT repo root with:
#     sbatch slurm/run_cgw_localization.sbatch
#
# Two modes:
#   MODE=sweep   (default) — run a built-in anchor-count sweep (0, 1, ..., K
#                 anchors); produces per-config skymaps and a 1-D scaling plot.
#   MODE=generate           — run a single anchor configuration; uses the
#                 ANCHOR_PULSARS variable as the explicit anchor list.
#
# Override at submit:
#   sbatch --export=ALL,OCARINA=ocarina,NSIDE=8 slurm/run_cgw_localization.sbatch
#   sbatch --export=ALL,MODE=generate,ANCHOR_PULSARS="J0437-4715 J0030+0451" \
#          slurm/run_cgw_localization.sbatch
#
# Before first submission:
#   1. Replace <TODO_FILL_IN> above with your account (`my_slurm_accounts`).
#   2. Build the overlay once (shared with the distance scan):
#        srun --cpus-per-task=2 --mem=10G --time=2:00:00 --pty bash
#        bash slurm/build_overlay.sh
#   3. Stage the ocarina dataset (small synthetic par/tim set; ~63 MB):
#        bash slurm/stage_ocarina.sh <netid>

set -euo pipefail

echo "[sbatch] Host: $(hostname)"
echo "[sbatch] Job:  ${SLURM_JOB_ID}"
printenv | grep -E '^SLURM_' | sort

module purge

# Don't preallocate the GPU (matches local behavior; avoids autotuner OOM).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Which staged dataset, which mode, knobs. Override at submit.
OCARINA="${OCARINA:-ocarina}"
MODE="${MODE:-sweep}"
NSIDE="${NSIDE:-4}"           # nside=4 -> 192 pixels; nside=8 -> 768 pixels
SNR_TARGET="${SNR_TARGET:-20.0}"
PIXEL_CHUNK="${PIXEL_CHUNK:-16}"  # smaller than distance map; Hessian tape is heavier
# Explicit anchor list for MODE=generate (space-separated PSR names).
ANCHOR_PULSARS="${ANCHOR_PULSARS:-J0437-4715 J0030+0451}"
# Use --full to run on the entire ocarina array instead of SMOKE_SUBSET.
FULL="${FULL:-0}"

if [ "${FULL}" = "1" ]; then
    FULL_FLAG="--full"
else
    FULL_FLAG=""
fi

export JAXPINT_OCARINA_DIR="${JAXPINT_OCARINA_DIR:-/scratch/${USER}/jaxpint-data/${OCARINA}}"

OVERLAY="/scratch/${USER}/jaxpint/overlay-15GB-500K.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

OUT_DIR="/scratch/${USER}/jaxpint-out/cgw-loc-${OCARINA}-${MODE}-${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

for path in "${OVERLAY}" "${IMAGE}" "${PROJECT_DIR}/examples/cgw_localization_skymap.py" "${JAXPINT_OCARINA_DIR}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[sbatch] ERROR: required path missing: ${path}" >&2
        exit 1
    fi
done

echo "[sbatch] OVERLAY        = ${OVERLAY}"
echo "[sbatch] IMAGE          = ${IMAGE}"
echo "[sbatch] PROJECT_DIR    = ${PROJECT_DIR}"
echo "[sbatch] OCARINA_DIR    = ${JAXPINT_OCARINA_DIR}"
echo "[sbatch] MODE           = ${MODE}"
echo "[sbatch] NSIDE          = ${NSIDE}"
echo "[sbatch] SNR_TARGET     = ${SNR_TARGET}"
echo "[sbatch] PIXEL_CHUNK    = ${PIXEL_CHUNK}"
echo "[sbatch] FULL           = ${FULL}"
echo "[sbatch] ANCHOR_PULSARS = ${ANCHOR_PULSARS}"
echo "[sbatch] OUT_DIR        = ${OUT_DIR}"

nvidia-smi || true

if [ "${MODE}" = "sweep" ]; then
    CLI="python examples/cgw_localization_skymap.py sweep \
        --out-dir '${OUT_DIR}' \
        --nside '${NSIDE}' \
        --snr '${SNR_TARGET}' \
        --pixel-chunk '${PIXEL_CHUNK}'"
elif [ "${MODE}" = "generate" ]; then
    OUT_FILE="${OUT_DIR}/cgw_loc_single.npz"
    CLI="python examples/cgw_localization_skymap.py generate \
        --output '${OUT_FILE}' \
        --nside '${NSIDE}' \
        --snr '${SNR_TARGET}' \
        --pixel-chunk '${PIXEL_CHUNK}' \
        --anchor-pulsars ${ANCHOR_PULSARS} \
        ${FULL_FLAG}"
else
    echo "[sbatch] ERROR: unknown MODE='${MODE}' (expected 'sweep' or 'generate')" >&2
    exit 1
fi

srun singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${OUT_DIR}:${OUT_DIR}" \
    --bind "${JAXPINT_OCARINA_DIR}:${JAXPINT_OCARINA_DIR}" \
    --env "JAXPINT_OCARINA_DIR=${JAXPINT_OCARINA_DIR}" \
    --env "XLA_PYTHON_CLIENT_PREALLOCATE=false" \
    --env "PYTHONHTTPSVERIFY=0" \
    "${IMAGE}" \
    /bin/bash -c "source /ext3/env.sh && cd '${PROJECT_DIR}' && ${CLI}"

echo "[sbatch] Done. Outputs in ${OUT_DIR}"
