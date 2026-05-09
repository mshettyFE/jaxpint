#!/bin/bash
#SBATCH --account=<TODO_FILL_IN>
#SBATCH --job-name=jaxpint-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Submit from the JaxPINT repo root with:
#     sbatch slurm/run_distance_scan.sbatch
#
# Before first submission:
#   1. Replace <TODO_FILL_IN> above with your account
#      (run `my_slurm_accounts` on Torch to discover yours).
#   2. Run `bash slurm/fetch_data.sh` to stage NANOGrav data on /scratch.
#   3. In an interactive session, run `bash slurm/build_overlay.sh` to build
#      the overlay (see that script's header for the srun command).

set -euo pipefail

echo "[sbatch] Host: $(hostname)"
echo "[sbatch] Job:  ${SLURM_JOB_ID}"
printenv | grep -E '^SLURM_' | sort

module purge

# Don't preallocate the GPU — XLA otherwise grabs ~75 % up front and the
# autotuner workspace fragments. Same setting the script uses locally.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Where the script reads par/tim files from.
export JAXPINT_DATA_DIR="/scratch/${USER}/jaxpint-data/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband"

OVERLAY="/scratch/${USER}/jaxpint/overlay-15GB-500K.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

OUT_DIR="/scratch/${USER}/jaxpint-out"
OUT_FILE="${OUT_DIR}/scan-${SLURM_JOB_ID}.npz"
mkdir -p "${OUT_DIR}"

# Sanity-check the artefacts we depend on; surface clear errors early.
for path in "${OVERLAY}" "${IMAGE}" "${PROJECT_DIR}/examples/nanograv_two_pulsar_distance_scan.py" "${JAXPINT_DATA_DIR}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[sbatch] ERROR: required path missing: ${path}" >&2
        exit 1
    fi
done

echo "[sbatch] OVERLAY     = ${OVERLAY}"
echo "[sbatch] IMAGE       = ${IMAGE}"
echo "[sbatch] PROJECT_DIR = ${PROJECT_DIR}"
echo "[sbatch] DATA_DIR    = ${JAXPINT_DATA_DIR}"
echo "[sbatch] OUT_FILE    = ${OUT_FILE}"

nvidia-smi || true

srun singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "${OUT_DIR}:${OUT_DIR}" \
    --bind "${JAXPINT_DATA_DIR}:${JAXPINT_DATA_DIR}" \
    --env "JAXPINT_DATA_DIR=${JAXPINT_DATA_DIR}" \
    --env "XLA_PYTHON_CLIENT_PREALLOCATE=false" \
    "${IMAGE}" \
    /bin/bash -c "source /ext3/env.sh && cd '${PROJECT_DIR}' && python examples/nanograv_two_pulsar_distance_scan.py generate --output '${OUT_FILE}' --profile"

echo "[sbatch] Done. Output at ${OUT_FILE}"
