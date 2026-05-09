# Running JaxPINT on NYU Torch HPC

This bundle runs `examples/nanograv_two_pulsar_distance_scan.py` on Torch
inside an Apptainer container, using an ext3 overlay for the Python env.
The local `.venv` is **not** shipped — the env is rebuilt from `uv.lock`
inside the overlay.

## One-time setup

1. **Clone the repo onto Torch.** Anywhere readable from compute nodes works;
   `/scratch` is recommended:

   ```sh
   ssh <netid>@torch.hpc.nyu.edu
   cd /scratch/$USER && git clone <this repo> jaxpint && cd jaxpint/JaxPINT
   ```

2. **Stage the NANOGrav 15-yr data** (login node, ~few GB download):

   ```sh
   bash slurm/fetch_data.sh
   ```

   Lays the dataset down at
   `/scratch/$USER/jaxpint-data/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband/`.

3. **Build the Apptainer overlay** with the JaxPINT env. Login nodes are
   capped at 2 GB RAM, so this **must** run inside an interactive session:

   ```sh
   srun --cpus-per-task=2 --mem=10G --time=2:00:00 --pty bash
   bash slurm/build_overlay.sh
   exit
   ```

   This populates `/scratch/$USER/jaxpint/overlay-15GB-500K.ext3` with `uv`
   and the project venv (`/ext3/venv`), and installs `/ext3/env.sh` for
   activation. Re-runs are no-ops once the overlay exists.

4. **Set your SLURM account** in `slurm/run_distance_scan.sbatch`. Find your
   account name with `my_slurm_accounts`, then replace `<TODO_FILL_IN>` in
   the `#SBATCH --account=` line.

## Submitting

From the JaxPINT repo root:

```sh
sbatch slurm/run_distance_scan.sbatch
squeue --me
tail -f slurm-<JOBID>.out
```

Output lands at `/scratch/$USER/jaxpint-out/scan-<JOBID>.npz`. Pull it back
to your laptop with `scp` or `rsync` against `dtn.torch.hpc.nyu.edu` and
run the `plot` subcommand locally if you want figures.

## What the sbatch script does

- Requests one GPU (`--gres=gpu:1`), 4 CPUs, 128 GB RAM, 4 h walltime.
  GPU type unconstrained — H100, H200, and L40S all comfortably fit the
  ~2 GB JIT-lowering footprint that OOMs your laptop.
- Sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (matches local behavior).
- Sets `JAXPINT_DATA_DIR=…/narrowband` so the script reads from the staged
  scratch path instead of its hardcoded `/home/hector/...` default.
- `apptainer exec --nv --overlay …:ro` runs the container with GPU drivers
  injected and the env mounted read-only.
- Runs the `generate` subcommand only — pure GPU compute, no matplotlib.
  Plot from the `.npz` afterwards.

## Caveats

- **`/scratch` purges after 60 days of no access.** If you go quiet that
  long, `tar` the data dir and the overlay onto `/archive` (2 TB, backed
  up) and re-stage on return.
- **Account is required.** Without an active allocation, sbatch rejects
  with "Invalid account". Request one via the HPC project portal at
  https://services.rt.nyu.edu/.
- **CUDA fallback.** The lockfile pins `jax[cuda13]`. JAX bundles its own
  user-space CUDA libs via that extra and Torch has fresh enough drivers
  for H200, so it should Just Work. If JAX can't see the GPU, re-enter
  the overlay rw and try `uv pip install --reinstall "jax[cuda12]==0.9.0"`.
