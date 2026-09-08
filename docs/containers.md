# NVIDIA GPU containers with Docker and Apptainer

This image targets NVIDIA GPU inference and defaults to `--device cuda`.

On an HPC machine with only Apptainer, build `containers/root.def` directly from
your checkout. Docker, BuildKit, and publishing to Docker Hub are not required.
For machines with Docker, `Dockerfile` provides the corresponding image and can
also be converted to SIF. Both recipes contain Python 3.12, root,
the dependencies in `uv.lock`, grammar support, pytest, ruff, Bash, and Git. It
installs root as a package and keeps a source snapshot for testing. No model
weights, host Git history, local configs, or credentials are baked in.

Commands below use Bash and start at the repository root unless stated otherwise.
Docker or Apptainer must already be installed on the relevant machine. Docker
Desktop provides a Linux VM on macOS and Windows; use WSL2 for these commands on
Windows. Apptainer commands below run on Linux.

## Choose the target machine

| Machine | Image platform | Runtime/device |
| --- | --- | --- |
| NVIDIA Linux server with Intel/AMD CPU | `linux/amd64` | Docker `--gpus all` or Apptainer `--nv`, `--device cuda` |
| NVIDIA Linux server with ARM CPU | `linux/arm64` | Same flags; verify support for the particular GPU and driver |
| Machine without an NVIDIA GPU | Match destination when building | Image building and unit tests only; default inference requires CUDA |

An amd64 image is not a native ARM image. Cross-building with Docker emulation is
possible but slower, especially because the build runs tests. A SIF is also tied
to its architecture. Containers on a Mac run Linux and do not expose Apple's MPS
backend; use the native installation when you want MPS.

The locked Linux PyTorch distribution includes NVIDIA dependencies even on a CPU
machine, so allow several GB for the image and build cache. The current lock uses
PyTorch 2.14.0 and CUDA 13 packages. A GPU host needs a driver compatible with the
CUDA runtime inside the image and a GPU supported by that PyTorch wheel. The GPU
check below reports the runtime actually installed. This recipe does not install
host drivers, vLLM, SGLang, Ollama, or ROCm.

## Build and develop using only Apptainer

The shared launcher detects Apptainer or Docker and prefers Apptainer when both
are installed. Set `ROOT_CONTAINER_RUNTIME=apptainer` or
`ROOT_CONTAINER_RUNTIME=docker` to select explicitly.

For a single command that builds the image if missing, tests it, and starts the
NVIDIA GPU terminal, run inside your GPU allocation:

```bash
./scripts/container.sh
```

You can also pass a prompt or root CLI options:

```bash
./scripts/container.sh --agent calc "What is 17 times 23, plus 4?"
```

The script uses the checkout as `/workspace` and reuses your host `HF_HOME`, or
`${XDG_CACHE_HOME:-$HOME/.cache}/huggingface` when `HF_HOME` is unset. It creates no
separate workspace or checkout-local model cache. An existing `root.sif` is reused;
its installed source snapshot is what runs. After changing source or dependencies,
use `ROOT_SIF=root-next.sif ./scripts/container.sh` to build a new snapshot, or use
the source-mount development commands below. The script preserves the scheduler's
`CUDA_VISIBLE_DEVICES` assignment. For sites that require building outside GPU
allocations, build separately first using the command below, then run the script
inside your allocation.

From the repository root on the HPC machine:

```bash
apptainer build --fakeroot root.sif containers/root.def
apptainer test --cleanenv root.sif
```

The definition copies your current source and tests, installs dependencies from
`uv.lock`, and runs the offline unit tests during the build. `apptainer test`
repeats those checks against the saved image. No GPU allocation or model download
is needed for these unit tests; inference and model evaluations require a GPU.

The recipe's `Bootstrap: docker` means **Apptainer downloads an OCI base image
directly from a registry**. It does not invoke Docker or need a Docker daemon.
The base images come from Docker Hub and GHCR; apt and Python dependencies also
need network access during a fresh build. You never need to upload your image.
See [Apptainer definition files](https://apptainer.org/docs/user/latest/definition_files.html).

Your site must permit unprivileged builds/fakeroot. `--fakeroot` does not grant
host root access; support depends on the site's user-namespace and fakeroot
configuration. If the site prohibits builds, an administrator must enable them
or provide a permitted build node. You can still test source edits using an
existing SIF. See [Apptainer build requirements](https://apptainer.org/docs/user/latest/build_a_container.html).

Build on a node where your site permits package downloads and image construction.
For large builds, put temporary files and the base-image cache on suitable local
scratch, replacing the paths below with directories available on that node:

```bash
mkdir -p /path/to/scratch/root-build-tmp /path/to/scratch/root-image-cache
export APPTAINER_TMPDIR=/path/to/scratch/root-build-tmp
export APPTAINER_CACHEDIR=/path/to/scratch/root-image-cache
apptainer build --fakeroot root.sif containers/root.def
```

### Run on an allocated NVIDIA GPU

First verify that the allocation and host driver expose CUDA:

```bash
apptainer exec --cleanenv --nv root.sif python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

If the scheduler sets `CUDA_VISIBLE_DEVICES`, preserve its assignment through
`--cleanenv` by setting `export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"`
inside that allocation before running Apptainer. Do not replace the scheduler's
assignment with a manually chosen GPU number.

Run the built source with persistent model cache, configs, and outputs:

```bash
mkdir -p container-work .cache/huggingface
apptainer run --cleanenv --nv \
  --bind "$PWD/container-work:/workspace" \
  --bind "$PWD/.cache/huggingface:/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --pwd /workspace \
  root.sif "What is 17 times 23, plus 4?"
```

Omit the prompt for the interactive terminal. The native SIF's runscript selects
the Transformers engine and CUDA. The first inference downloads model weights;
subsequent runs reuse the mounted cache.

### Edit, test, and run without rebuilding

Keep editing the checkout normally, then test those edits using the SIF's locked
dependencies. The runner copies source and fixtures into temporary writable
storage, so neither the SIF nor the checkout needs to be modified by the tests:

```bash
apptainer exec --cleanenv \
  --bind "$PWD:/checkout:ro" \
  root.sif root-container-test /checkout
```

To run the edited source on the GPU, explicitly select it with `PYTHONPATH`:

```bash
apptainer exec --cleanenv --nv \
  --bind "$PWD:/checkout:ro" \
  --bind "$PWD/container-work:/workspace" \
  --bind "$PWD/.cache/huggingface:/cache/huggingface" \
  --env PYTHONPATH=/checkout/src \
  --env HF_HOME=/cache/huggingface \
  --pwd /workspace \
  root.sif root --engine transformers --device cuda
```

For model evaluations of the edited source, keep those options and replace the
command after `root.sif` with
`root-eval --device cuda --agent calc --output-dir runs/eval-calc`.
Outputs appear under `container-work/runs/` on the host.

Rebuild after changing dependencies, or when you want to capture your source edits
in a portable image. If dependencies changed, update and review `uv.lock` first.
Build a new filename to retain the previous working image:

```bash
apptainer build --fakeroot root-next.sif containers/root.def
apptainer test --cleanenv root-next.sif
```

Both recipes use the same Python/uv versions and lockfile. They are separate
build recipes, not byte-identical images; transfer a specific SIF when you need
the exact same built environment on another compatible machine.

## Build and test with Docker

For the same build/test/run workflow as `scripts/container.sh`, use:

```bash
ROOT_CONTAINER_RUNTIME=docker ./scripts/container.sh
ROOT_CONTAINER_RUNTIME=docker ./scripts/container.sh --agent calc "What is 17 times 23, plus 4?"
```

The Docker daemon must be running and configured with NVIDIA Container Toolkit.
The script builds `root:local` if absent, runs the offline unit suite, and starts
CUDA inference. It mounts the checkout as `/workspace` and reuses host `HF_HOME`,
or `${XDG_CACHE_HOME:-$HOME/.cache}/huggingface`. Output files belong to your host
UID/GID, and prompts and root CLI options are forwarded unchanged.

Set `ROOT_DOCKER_IMAGE=root:next ROOT_CONTAINER_RUNTIME=docker ./scripts/container.sh` to build a fresh source
snapshot under a new tag after edits. Existing tags are reused, not rebuilt
automatically. `ROOT_DOCKER_GPUS` defaults to `all`; set it to a Docker GPU request
such as `device=0` to restrict devices. On managed systems, use only GPUs assigned
to you; Docker GPU selection is separate from the Apptainer scheduler forwarding.

Build for the current machine:

```bash
docker build --tag root:local .
```

The build installs dependencies with `uv sync --locked --extra dev --no-editable`,
checks the installed CLI, and runs the complete unit suite as an unprivileged
user. A failing test fails the build. The tests run offline without downloading a
model, in a temporary repository with writable fixtures. Dependency installation
still needs network access on the first build.

To build for an Intel/AMD server from another architecture:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag root:amd64 .
```

Repeat the built-in checks without any host mounts:

```bash
docker run --rm --network none root:local root --version
docker run --rm --network none root:local root-container-test
docker run --rm root:local ruff check --no-cache /opt/root/src /opt/root/tests
```

Lint runs separately from the build so existing lint issues are visible without
preventing a runtime image from being created. Select individual tests by giving
the source directory followed by pytest arguments:

```bash
docker run --rm root:local root-container-test /opt/root tests/test_tools.py
```

The Python versions come from `uv.lock`; uv's version is pinned in the Dockerfile.
The Python base tag and Debian apt packages can change between builds. For the
same environment on every machine, transfer the built image/SIF, or pull by image
digest, rather than rebuilding separately.

## Run root with Docker

Create persistent directories once. These are the only host directories needed
for normal inference; run outputs and local configs live in `container-work`.

```bash
mkdir -p container-work .cache/huggingface
```

Interactive NVIDIA GPU session (run the CUDA check below first):

```bash
docker run --rm -it \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/container-work,dst=/workspace" \
  --mount "type=bind,src=$PWD/.cache/huggingface,dst=/cache/huggingface" \
  root:local root --engine transformers --device cuda
```

One prompt, with a saved result:

```bash
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/container-work,dst=/workspace" \
  --mount "type=bind,src=$PWD/.cache/huggingface,dst=/cache/huggingface" \
  root:local root --engine transformers --device cuda \
  --agent calc --output-dir runs/smoke "What is 17 times 23, plus 4?"
```

The first inference downloads the default model into the mounted Hugging Face
cache. Later runs reuse it. Add `--model LiquidAI/LFM2.5-350M` to select it explicitly,
or use another configured model. Add `--env HF_TOKEN` before the image name for a
gated model, after setting that variable in your host shell; do not put tokens in
the Dockerfile or build arguments.

To initialize editable configs, use the same mounts and replace the command after
`root:local` with `root --init`. Edit `container-work/configs/` on the host. File
tools operate in `/workspace`, and writes persist on the host. The image defaults
to UID 1000; `--user` above makes bind-mounted outputs belong to your account.
Container-local terminal history under `/tmp` is discarded when the container exits.

### NVIDIA GPU

The Docker host needs the NVIDIA driver and NVIDIA Container Toolkit configured.
Check device access before downloading a model:

```bash
docker run --rm --gpus all root:local python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Run this check on every destination machine before inference. The inference
commands already request NVIDIA passthrough and CUDA explicitly, so they fail
instead of silently falling back to CPU when CUDA is unavailable. You do not
need a GPU during the build or for the unit tests. On a cluster, request a GPU
through the scheduler before running the check or inference.

### Model evaluations

Unit tests check software behavior. `root-eval` loads a real model and scores tool
choice and answer content, so it takes longer and is a separate step:

```bash
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/container-work,dst=/workspace" \
  --mount "type=bind,src=$PWD/.cache/huggingface,dst=/cache/huggingface" \
  root:local root-eval --device cuda --agent calc --output-dir runs/eval-calc
```

Remove `--agent calc` for all agents. Add `--min-accuracy 0.8` to make accuracy
below 80% return a failing exit status, choosing a threshold appropriate to your
model. Results are written to `container-work/runs/eval-calc/evals.json`.
Packaged evals include their own workspace fixtures; custom eval configs can set
their own paths.

## Move the image to another machine

No registry is required. Build for the destination's architecture, then export:

```bash
docker save --output root-amd64.tar root:amd64
scp root-amd64.tar user@server:/path/to/images/
```

On a Docker destination:

```bash
docker load --input /path/to/images/root-amd64.tar
docker run --rm root:amd64 root-container-test
```

Use `root:amd64` in subsequent commands. You can instead tag and push the image
to a registry you control, then pull it on the other machines. No image is
published automatically by this repository.

## Apptainer: use the same image on a cluster

This section covers optional Docker-image conversion. For an Apptainer-only HPC
build and development workflow, use `containers/root.def` above.

No Docker Hub upload is required. If your Apptainer version supports the
`dockerfile` bootstrap agent and a BuildKit daemon is configured, build directly
from the repository on the destination architecture:

```bash
apptainer build root.sif dockerfile:.
```

This requires BuildKit support in your installation; see the
[Apptainer Dockerfile builder documentation](https://apptainer.org/docs/user/main/appendix.html#dockerfile-bootstrap-agent).
Otherwise, use either conversion method below.

On the Linux machine with Apptainer, convert the transferred Docker archive:

```bash
apptainer build root.sif docker-archive:/path/to/images/root-amd64.tar
```

Alternatively, on Linux with access to the Docker daemon that built the image:

```bash
apptainer build root.sif docker-daemon:root:local
```

For a published image, use `apptainer pull root.sif docker://REGISTRY/OWNER/root:TAG`,
replacing the registry, owner, and tag. You can copy the resulting `root.sif` to
other machines with the same architecture. Site policy controls whether image
conversion is allowed on login nodes; it can be done on another Linux machine
and the SIF transferred instead.

Run the embedded tests, without a checkout or writable image:

```bash
apptainer exec --cleanenv root.sif root --version
apptainer exec --cleanenv root.sif root-container-test
```

From a directory containing the SIF, create the mounts and run a GPU prompt (inside a GPU allocation, after the CUDA check below):

```bash
mkdir -p container-work .cache/huggingface
apptainer exec --cleanenv --nv \
  --bind "$PWD/container-work:/workspace" \
  --bind "$PWD/.cache/huggingface:/cache/huggingface" \
  --env HF_HOME=/cache/huggingface \
  --pwd /workspace \
  root.sif root --engine transformers --device cuda \
  --agent calc --output-dir runs/smoke "What is 17 times 23, plus 4?"
```

Omit the prompt and `--agent`/`--output-dir` for an interactive session. Apptainer
runs as your host user, so no Docker-style `--user` is needed. `--cleanenv` avoids
inheriting host Python environment settings. Apptainer can also bind host home
and other paths by default, according to site configuration; the explicit binds
above specify the persistent model cache and working directory.

GPU check (inside your GPU allocation):

```bash
apptainer exec --cleanenv --nv root.sif python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The inference command above already includes `--nv` and `--device cuda`.
For evaluations, keep the mounts and replace the command after `root.sif` with
`root-eval --device cuda --agent calc --output-dir runs/eval-calc`.
For gated models, set `export APPTAINERENV_HF_TOKEN="$HF_TOKEN"` in the host shell
before running Apptainer; this explicitly forwards the token with `--cleanenv`.
If temporary storage is too small, set host `APPTAINER_TMPDIR` to a
writable scratch directory before conversion, and bind suitable scratch storage
for model downloads and outputs.

## Offline machines

Transfer the image/SIF and a populated `.cache/huggingface` directory. Warm the
cache with an actual inference using each intended model on a connected machine
first; the container contains no weights. Preserve the cache directory structure
and symlinks when copying it. On Docker, add `--network none` and
`--env HF_HUB_OFFLINE=1`. On Apptainer, add `--env HF_HUB_OFFLINE=1`; this flag
prevents Hugging Face downloads, but does not disable networking for other tools.

## Test edits without rebuilding

Bind a checkout read-only and run its tests against a temporary copy of its source
using the image's installed dependencies:

```bash
docker run --rm --network none \
  --mount "type=bind,src=$PWD,dst=/checkout,readonly" \
  root:local root-container-test /checkout
```

```bash
apptainer exec --cleanenv \
  --bind "$PWD:/checkout:ro" \
  /path/to/root.sif root-container-test /checkout
```

Rebuild when `pyproject.toml` or `uv.lock` changes. To run edited code directly,
mount the checkout and set `PYTHONPATH=/checkout/src` in the container environment;
otherwise `root` runs the installed snapshot. Do not run `uv sync` inside a SIF:
its installed environment is read-only.

## Troubleshooting and verification status

- Build says the lock is stale: run `uv lock` in the checkout, review the update,
  and rebuild. Do not remove `--locked` to hide dependency drift.
- CUDA is unavailable: check the host driver, scheduler allocation, device
  passthrough flag, and the CUDA version printed by the diagnostic above. The
  default command requires CUDA; resolve the host/runtime compatibility before
  running GPU inference.
- Permission errors: create mount directories as your host user; use Docker's
  `--user` option. Keep caches and outputs on writable binds with Apptainer.
- Source changes have no effect: rebuild, or explicitly set the mounted source
  path as described above.
- A remote inference engine is unreachable: `localhost` inside Docker refers to
  the container. Pass `--engine-url` with a reachable server address and select
  the engine explicitly. This image contains the client, not those servers.

Docker and Apptainer were unavailable on the development machine when this recipe
was added. The recipe therefore still needs its first actual image build and
NVIDIA GPU smoke run on a host with a container runtime. The build-time test step
and commands above are the acceptance checks; an image tag alone does not prove
GPU or model accuracy compatibility.

Reference behavior: [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/),
[Docker platform builds](https://docs.docker.com/build/building/multi-platform/),
[Docker GPU access](https://docs.docker.com/engine/containers/gpu/),
[Apptainer Docker/OCI support](https://apptainer.org/docs/user/latest/docker_and_oci.html),
[Apptainer archive conversion](https://apptainer.org/docs/user/main/appendix.html),
and [Apptainer GPU support](https://apptainer.org/docs/user/main/gpu.html).
