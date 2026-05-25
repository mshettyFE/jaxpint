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
#     /scratch/<netid>/jaxpint-data/ocarina/{par,tim}/
# which is the path baked into slurm/run_cgw_skymap.sbatch via
# $JAXPINT_OCARINA_DIR.

set -euo pipefail

NETID="${1:?usage: bash slurm/stage_ocarina.sh <netid> [local_ocarina_dir]}"

# Default: ocarina lives in the parent of the JaxPINT checkout (jax_pint/ocarina).
DEFAULT_OCARINA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/ocarina"
LOCAL_OCARINA="${2:-${DEFAULT_OCARINA}}"

DEST_HOST="dtn.torch.hpc.nyu.edu"      # data-transfer node (outbound rsync ok)
DEST="/scratch/${NETID}/jaxpint-data/ocarina"

if [[ ! -d "${LOCAL_OCARINA}/par" || ! -d "${LOCAL_OCARINA}/tim" ]]; then
    echo "ERROR: ${LOCAL_OCARINA} does not look like an ocarina dir" >&2
    echo "       (expected par/ and tim/ subdirectories)." >&2
    exit 1
fi

N_PAR=$(find "${LOCAL_OCARINA}/par" -name '*.par' | wc -l | tr -d ' ')
N_TIM=$(find "${LOCAL_OCARINA}/tim" -name '*.tim' | wc -l | tr -d ' ')
echo "[stage_ocarina] Local:  ${LOCAL_OCARINA}  (${N_PAR} par, ${N_TIM} tim)"
echo "[stage_ocarina] Remote: ${NETID}@${DEST_HOST}:${DEST}"

ssh "${NETID}@${DEST_HOST}" "mkdir -p '${DEST}'"
rsync -avh --progress "${LOCAL_OCARINA}/par" "${LOCAL_OCARINA}/tim" \
    "${NETID}@${DEST_HOST}:${DEST}/"

echo
echo "[stage_ocarina] Done. On Torch, the sbatch will read:"
echo "    JAXPINT_OCARINA_DIR=${DEST}"
