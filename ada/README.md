# Ada

Running this pipeline on IIIT-H's cluster without changing what the pipeline is.
`python build.py` is the interface on a laptop and stays the interface here;
everything in this directory is placement.

    ada/sync.sh code            push the repo to Ada
    ada/sync.sh data            push the raw archives (~3.4 GB)
    sbatch ada/env.sbatch       build the Python environment, once
    sbatch ada/build.sbatch     run the pipeline
    ada/sync.sh pull            bring artifacts back

## What was measured, so a later run does not have to ask

The user guide leaves several things open and gets two wrong for this account.
All of the below was probed on **2026-08-21** with `srun` and `sinfo` rather
than read off the guide.

| | guide / plan assumed | measured |
|---|---|---|
| Accounts | maybe `irel` (120 CPU, 12 GPU) | **`research` only**, QoS `medium` |
| QoS ceiling | 40 CPU × 3000 MB | `cpu=40, gres/gpu=4, mem=125000M`, 4-day wall, 4 running / 8 queued |
| Compute OS | Ubuntu 18.04, glibc **2.27** | **Ubuntu 22.04.5, glibc 2.35**, Python 3.10.12 |
| Login node | — | **CentOS 7, glibc 2.17** — *not the same as compute* |
| 1 GPU : 10 CPU ratio | might force 4 GPUs to get 40 CPUs | **does not bind zero-GPU jobs** |
| `/home` | 25 GB | **30 GB** (`/home2`, NFS) |

**The wheel worry was unfounded.** The plan expected glibc 2.27 to refuse
`manylinux_2_28` wheels. Compute nodes run **2.35** and accept them. The real
hazard is the opposite one and is easy to miss: the **login node is glibc
2.17**, so an environment resolved there is not the environment a job runs
under. `ada/env.sbatch` therefore builds on a compute node.

**No email to the admins was needed about the GPU ratio.** Requesting
`-c 20 --mem=60G` and `-c 32 --mem=100G` with **no GPU** both allocate. `-c 40`
does not — not policy, but topology: a node has exactly 40 cores and one is
rarely wholly free. So **32 CPUs / 100 GB is the practical ceiling**, against
Phase 8's projected 40–60 GB.

**Two partitions are closed to `research`** and are worth asking for:

- **`u22-cpu`** — 6 nodes, 128 GB, **no GPUs**. Eight of this pipeline's nine
  stages never touch a GPU, so running them here would stop us occupying GPU
  nodes to compute BM25 and bootstrap intervals. This is the better request.
- **`ihub`** — 12 nodes, 257 GB each. Only worth asking for if Phase 8 needs
  more than 100 GB.

**`/share1/dataset` has no MIND or EB-NeRD.** It holds a lot of vision and
speech corpora and was worth checking — the guide says public datasets may be
pre-staged — but these two are not there. `ada/sync.sh data` pushes the
archives instead of re-downloading them.

## The filesystem is the whole design

From a **compute node**, checked rather than assumed:

```
VISIBLE  /home        30 GB    persistent      code, environment, results
VISIBLE  /scratch      2 TB    node-local, purged after 7 days
VISIBLE  /ssd_scratch 960 GB   node-local, purged after 7 days
ABSENT   /share1     100 GB    login node only
ABSENT   /share2
```

The only shared filesystem a job can read is the smallest one, and the roomy one
cannot be reached from a job at all. That is why the project is split three ways:

- **`/scratch`** takes `data/` and `feature_store/` — big, rebuildable, and
  gone next week, which is exactly what they are.
- **`$HOME/vub-news-run`** takes `artifacts/`, `predictions/` and
  `.checkpoints/` — small and worth keeping, and reachable from the job that
  writes them.
- **`/share1/$USER`** is the archive, filled by `ada/sync.sh park` from the
  login node afterwards, because no job can write there.

Phase 1's relocatable root is what makes this a wrapper rather than a rewrite:
`VUB_NEWS_DATA_ROOT` moves the heavy pair and `VUB_NEWS_ROOT` moves the light
three, independently, and `build.py` never learns which machine it is on.

## What is not automated, and why

**`park` is a separate command.** It could have been the tail of
`build.sbatch`, except that /share1 is invisible from the compute node the job
runs on, so it physically cannot be. It runs on the login node or not at all.

**`env.sbatch` is not idempotent by accident.** It skips miniforge and the env
if they already exist, so re-running it only re-installs the pinned
requirements. Deleting `$HOME/envs/vub-news` is how you force a clean rebuild.

**The stage-out is a `trap`, and it does not mirror on failure.** `build.sbatch`
is `set -e`, so a build that raises used to abandon the rest of the script and
strand the feature store on node-local `/scratch` — which is how job 2674571
lost eight stages' worth of it to a seven-day purge window. It runs on the way
out now, whatever the exit status. What it drops on the failure path is
`--delete`: a crash part-way through `ingest` leaves a partial store on
`/scratch` while `/home` still holds the whole one, and mirroring the first
onto the second would faithfully destroy it. Stale files are recoverable, and
the next clean run deletes them; deleted ones are not. `SIGKILL` is still
`SIGKILL`, so a job that overruns its walltime loses `/scratch` regardless.
