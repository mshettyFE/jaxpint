#!/bin/bash
# build_overlay.sh — one-time setup of the Apptainer ext3 overlay containing
# uv + the JaxPINT env on Torch.
#
# MUST be run from inside an interactive SLURM session (login nodes are
# capped at 2 GB RAM; pip/uv installs will OOM there). Open one with:
#
#     srun --cpus-per-task=2 --mem=10G --time=2:00:00 --pty bash
#
# Then `cd` into the JaxPINT repo and run `bash slurm/build_overlay.sh`.

set -euo pipefail

# --- Host-side preflight ---------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: \$SLURM_JOB_ID is unset; you appear to be on a login node." >&2
    echo "       Run an interactive session first:" >&2
    echo "         srun --cpus-per-task=2 --mem=10G --time=2:00:00 --pty bash" >&2
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[build_overlay] Project directory: ${PROJECT_DIR}"

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" || ! -f "${PROJECT_DIR}/uv.lock" ]]; then
    echo "ERROR: ${PROJECT_DIR} does not look like a JaxPINT checkout" >&2
    echo "       (missing pyproject.toml or uv.lock)." >&2
    exit 1
fi

OVERLAY_DIR="/scratch/${USER}/jaxpint"
OVERLAY_TEMPLATE="/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz"
OVERLAY="${OVERLAY_DIR}/overlay-15GB-500K.ext3"
IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

mkdir -p "${OVERLAY_DIR}"

if [[ ! -f "${OVERLAY}" ]]; then
    echo "[build_overlay] Copying overlay template from ${OVERLAY_TEMPLATE}"
    cp -p "${OVERLAY_TEMPLATE}" "${OVERLAY_DIR}/"
    echo "[build_overlay] Decompressing"
    gunzip "${OVERLAY_DIR}/$(basename "${OVERLAY_TEMPLATE}")"
else
    echo "[build_overlay] Overlay already exists at ${OVERLAY}; reusing in :rw mode."
fi

if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: Apptainer image not found at ${IMAGE}." >&2
    echo "       Check /share/apps/images/ for available .sif files and edit this script." >&2
    exit 1
fi

# --- Inner build (inside the container, with the overlay mounted rw) -------
echo "[build_overlay] Entering container; running inner setup..."

singularity exec --fakeroot \
    --overlay "${OVERLAY}:rw" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${IMAGE}" \
    /bin/bash -c '
set -euo pipefail
echo "[inner] uname: $(uname -a)"

# Install uv into the overlay if missing
if [[ ! -x /ext3/uv/uv ]]; then
    echo "[inner] Installing uv to /ext3/uv"
    mkdir -p /ext3/uv
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/ext3/uv UV_NO_MODIFY_PATH=1 sh
fi
export PATH="/ext3/uv:${PATH}"
uv --version

# Sync the project venv into /ext3/venv from the lockfile.
export UV_PROJECT_ENVIRONMENT=/ext3/venv
cd "'"${PROJECT_DIR}"'"
echo "[inner] uv sync --extra cuda (slow step; downloads JAX + CUDA wheels)"
uv sync --extra cuda

# Activation helper sourced by sbatch scripts when the overlay is :ro.
cat > /ext3/env.sh <<EOF
#!/bin/bash
# Activate the JaxPINT env inside the Apptainer overlay.
unset -f which
export PATH="/ext3/venv/bin:/ext3/uv:\${PATH}"
EOF
chmod +x /ext3/env.sh

echo "[inner] Smoke test (CPU-only here; --nv only set at sbatch time)"
/ext3/venv/bin/python -c "import jax, jaxpint; print(\"jax\", jax.__version__); print(\"jaxpint OK\"); print(\"devices:\", jax.devices())"
'

echo
echo "[build_overlay] Done."
echo "[build_overlay] Overlay ready: ${OVERLAY}"
echo
echo "Next: edit slurm/run_distance_scan.sbatch to set --account=<your-account>,"
echo "      then submit with:  sbatch slurm/run_distance_scan.sbatch"
