#!/bin/bash
# fetch_data.sh — one-time NANOGrav 15-yr download from Zenodo onto Torch.
#
# Run on the Torch login node (login nodes have outbound HTTPS).
# Re-run is a no-op if the extracted tree is already present.
#
# Lays down:
#   /scratch/$USER/jaxpint-data/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband/...
# which is the path baked into slurm/run_distance_scan.sbatch via $JAXPINT_DATA_DIR.

set -euo pipefail

DATA_ROOT="/scratch/${USER}/jaxpint-data"
TARBALL="NANOGrav15yr_PulsarTiming_v2.0.0.tar.gz"
ZENODO_URL="https://zenodo.org/records/8423265/files/${TARBALL}"
EXTRACTED_DIR="${DATA_ROOT}/NANOGrav15yr_PulsarTiming_v2.0.0"
NARROWBAND_DIR="${EXTRACTED_DIR}/narrowband"

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

if [[ -d "${NARROWBAND_DIR}" ]]; then
    echo "[fetch_data] ${NARROWBAND_DIR} already present; skipping."
    exit 0
fi

if [[ ! -f "${TARBALL}" ]]; then
    echo "[fetch_data] Downloading ${ZENODO_URL}"
    wget --no-verbose --show-progress "${ZENODO_URL}"
fi

echo "[fetch_data] Extracting ${TARBALL}"
tar -xzf "${TARBALL}"

echo "[fetch_data] Verifying contents"
ls -1 "${NARROWBAND_DIR}" | head -5
N_PARS=$(find "${NARROWBAND_DIR}" -name '*.par' | wc -l)
N_TIMS=$(find "${NARROWBAND_DIR}" -name '*.tim' | wc -l)
echo "[fetch_data] Found ${N_PARS} .par files and ${N_TIMS} .tim files."

echo
echo "[fetch_data] Done. Set in your sbatch:"
echo "    export JAXPINT_DATA_DIR=${NARROWBAND_DIR}"
