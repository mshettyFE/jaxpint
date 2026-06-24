#!/bin/bash
#SBATCH --account=<TODO_FILL_IN>
#SBATCH --job-name=jaxpint-cgw-incoherent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Real-mode Bayesian CW distance-reach sky map with the pulsar term marginalized
# over the pulsar DISTANCE (PX +/- k*sigma uniform prior). Runs
# examples/cgw_incoherent_distance_skymap.py inside the same Apptainer overlay as
# run_cgw_skymap.sbatch -- no new Python deps.
#
# Heavier than run_cgw_skymap.sbatch: it marginalizes the timing model and runs an
# autodiff (b,M) extraction for EVERY sky pixel of EVERY pulsar (rather than one
# coherent pass), so --time is bumped to 8h. Time a small `--nside 4` run first to
# calibrate before committing to the full nside=8.
#
# Submit from the JaxPINT repo root with:
#     sbatch slurm/run_cgw_incoherent.sbatch
# Override knobs at submit, e.g.:
#     sbatch --export=ALL,OCARINA=ocarina,NSIDE=4,PIXEL_CHUNK=4,KSIGMA=5,NPHASE=256 \
#            slurm/run_cgw_incoherent.sbatch
#
# Before first submission:
#   1. Replace <TODO_FILL_IN> above with your account (`my_slurm_accounts`).
#   2. Build the overlay once (shared with the other sky-map jobs):
#        srun --cpus-per-task=2 --mem=10G --time=2:00:00 --pty bash
#        bash slurm/build_overlay.sh
#   3. Stage the ocarina dataset onto /scratch. From your LAPTOP:
#        bash slurm/stage_ocarina.sh <netid>

set -euo pipefail

echo "[sbatch] Host: $(hostname)"
echo "[sbatch] Job:  ${SLURM_JOB_ID}"
printenv | grep -E '^SLURM_' | sort

module purge

# Don't preallocate the GPU (matches local behavior; avoids autotuner OOM).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Which staged dataset to run, plus the incoherent-UL fidelity knobs. Override at
# submit (see header). This map is always real-mode, fixed-orientation, and
# distance-marginalized, so there is no DATA_MODE/MARGINALIZE/PULSAR_TERM toggle.
OCARINA="${OCARINA:-ocarina}"
NSIDE="${NSIDE:-8}"            # HEALPix nside (npix = 12*nside^2; 8 -> 768 pixels)
PIXEL_CHUNK="${PIXEL_CHUNK:-8}"   # sky pixels vmapped per chunk (memory <-> speed)
KSIGMA="${KSIGMA:-5}"         # distance prior half-width in units of sigma_PX (L in [1/PX +/- k*sigma_L])
NPHASE="${NPHASE:-256}"       # pulsar-term phase grid points (flat-phase marginal)
EARTH_TERM="${EARTH_TERM:-0}"  # 1 -> method-matched Earth-term baseline (no pulsar term)
COHERENT_FRACTION="${COHERENT_FRACTION:-0}"   # fraction (best PX/sigma) with a tight distance prior (coherent)
COHERENT_SIGMA_PC="${COHERENT_SIGMA_PC:-0.1}" # tight-subset distance prior 1-sigma in pc (~0.01 localizes, ~1 ~ flat)

# Earth-term toggle: drops the pulsar term and routes the pure Earth signal
# through the SAME numerical UL, for a controlled earth-vs-incoherent comparison.
if [ "${EARTH_TERM}" = "1" ]; then
    EARTH_FLAG="--earth-term-only"
    EARTH_TAG="-earth"
else
    EARTH_FLAG=""
    EARTH_TAG=""
fi

# Ocarina par/tim directory staged by slurm/stage_ocarina.sh (mirrors the local
# dataset name). The driver reads this via $JAXPINT_OCARINA_DIR.
export JAXPINT_OCARINA_DIR="${JAXPINT_OCARINA_DIR:-/scratch/${USER}/jaxpint-data/${OCARINA}}"

OVERLAY="/scratch/${USER}/jaxpint/overlay-15GB-500K.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

OUT_DIR="/scratch/${USER}/jaxpint-out"
# Filename tag for the coherent subset (omit when fraction is 0).
case "${COHERENT_FRACTION}" in
    0|0.0|0.00|"") COH_TAG="" ;;
    *)             COH_TAG="-coh${COHERENT_FRACTION}-s${COHERENT_SIGMA_PC}pc" ;;
esac
OUT_FILE="${OUT_DIR}/cgw-incoherent-${OCARINA}-real${EARTH_TAG}${COH_TAG}-${SLURM_JOB_ID}.npz"
mkdir -p "${OUT_DIR}"

# Sanity-check the artefacts we depend on; surface clear errors early.
for path in "${OVERLAY}" "${IMAGE}" \
            "${PROJECT_DIR}/examples/cgw_incoherent_distance_skymap.py" \
            "${JAXPINT_OCARINA_DIR}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[sbatch] ERROR: required path missing: ${path}" >&2
        exit 1
    fi
done

echo "[sbatch] OVERLAY     = ${OVERLAY}"
echo "[sbatch] IMAGE       = ${IMAGE}"
echo "[sbatch] PROJECT_DIR = ${PROJECT_DIR}"
echo "[sbatch] OCARINA_DIR = ${JAXPINT_OCARINA_DIR}"
echo "[sbatch] NSIDE       = ${NSIDE}"
echo "[sbatch] PIXEL_CHUNK = ${PIXEL_CHUNK}"
echo "[sbatch] KSIGMA      = ${KSIGMA}"
echo "[sbatch] NPHASE      = ${NPHASE}"
echo "[sbatch] EARTH_TERM  = ${EARTH_TERM} (${EARTH_FLAG:-pulsar term: distance-marginalized})"
echo "[sbatch] COHERENT    = fraction ${COHERENT_FRACTION}, sigma ${COHERENT_SIGMA_PC} pc"
echo "[sbatch] OUT_FILE    = ${OUT_FILE}"

nvidia-smi || true

# `generate` only -- pure compute, no matplotlib. Plot from the .npz afterwards
# with examples/cgw_earth_vs_pulsar_distance.py. `--full` uses all pulsars.
srun singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${OUT_DIR}:${OUT_DIR}" \
    --bind "${JAXPINT_OCARINA_DIR}:${JAXPINT_OCARINA_DIR}" \
    --env "JAXPINT_OCARINA_DIR=${JAXPINT_OCARINA_DIR}" \
    --env "XLA_PYTHON_CLIENT_PREALLOCATE=false" \
    --env "PYTHONHTTPSVERIFY=0" \
    "${IMAGE}" \
    /bin/bash -c "source /ext3/env.sh && cd '${PROJECT_DIR}' && python examples/cgw_incoherent_distance_skymap.py generate --output '${OUT_FILE}' --nside '${NSIDE}' --full --pixel-chunk '${PIXEL_CHUNK}' --k '${KSIGMA}' --n-phase '${NPHASE}' --coherent-fraction '${COHERENT_FRACTION}' --coherent-sigma-pc '${COHERENT_SIGMA_PC}' ${EARTH_FLAG}"

echo "[sbatch] Done. Output at ${OUT_FILE}"
