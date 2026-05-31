#!/bin/bash
# stage_ocarina.sh — push the local ocarina par/tim dataset to Torch /scratch.
#
# Run this on your LAPTOP (the machine that has the ocarina/ directory), NOT on
# Torch. ocarina is the small synthetic dataset built by build_ocarina.py
# (stripped pars: astrometry + spindown + noise only); it is not on Zenodo, so
# we copy it directly rather than re-downloading.
#
# Usage:
#     bash slurm/stage_ocarina.sh <netid> [local_ocarina_dir]
#
# Lands the data at:
#     /scratch/<netid>/jaxpint-data/<dataset>/{par,tim}/  (+ seed.txt if present)
# where <dataset> mirrors the local directory name (e.g. ocarina, ocarina_2), so
# multiple seeds can be staged side by side. run_cgw_skymap.sbatch reads the
# chosen one via $JAXPINT_OCARINA_DIR.

set -euo pipefail

NETID="${1:?usage: bash slurm/stage_ocarina.sh <netid> [local_ocarina_dir]}"

# Default: ocarina_2 lives in the parent of the JaxPINT checkout (jax_pint/ocarina_2).
DEFAULT_OCARINA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/ocarina"
LOCAL_OCARINA="${2:-${DEFAULT_OCARINA}}"

# Mirror the local dataset's directory name on the remote so multiple datasets
# (e.g. ocarina, ocarina_2) can be staged side by side.
DATASET_NAME="$(basename "${LOCAL_OCARINA}")"

DEST_HOST="dtn.torch.hpc.nyu.edu"      # data-transfer node (outbound rsync ok)
DEST="/scratch/${NETID}/jaxpint-data/${DATASET_NAME}"

if [[ ! -d "${LOCAL_OCARINA}/par" || ! -d "${LOCAL_OCARINA}/tim" ]]; then
    echo "ERROR: ${LOCAL_OCARINA} does not look like an ocarina dir" >&2
    echo "       (expected par/ and tim/ subdirectories)." >&2
    exit 1
fi

N_PAR=$(find "${LOCAL_OCARINA}/par" -name '*.par' | wc -l | tr -d ' ')
N_TIM=$(find "${LOCAL_OCARINA}/tim" -name '*.tim' | wc -l | tr -d ' ')
echo "[stage_ocarina] Local:  ${LOCAL_OCARINA}  (${N_PAR} par, ${N_TIM} tim)"
echo "[stage_ocarina] Remote: ${NETID}@${DEST_HOST}:${DEST}"

# Always stage par/ and tim/; include seed.txt (build_ocarina's noise-seed
# provenance) when present so the remote dataset records which draw it is.
sources=("${LOCAL_OCARINA}/par" "${LOCAL_OCARINA}/tim")
if [[ -f "${LOCAL_OCARINA}/seed.txt" ]]; then
    sources+=("${LOCAL_OCARINA}/seed.txt")
else
    echo "[stage_ocarina] WARNING: no seed.txt in ${LOCAL_OCARINA} (staging par/tim only)." >&2
fi

ssh "${NETID}@${DEST_HOST}" "mkdir -p '${DEST}'"
rsync -avh --progress "${sources[@]}" "${NETID}@${DEST_HOST}:${DEST}/"

echo
echo "[stage_ocarina] Done. On Torch, the sbatch will read:"
echo "    JAXPINT_OCARINA_DIR=${DEST}"
