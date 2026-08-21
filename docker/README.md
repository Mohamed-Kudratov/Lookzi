# Pod image

Stopping a RunPod pod wipes the container disk, so every restart — and every
forced migration when a GPU is reclaimed — costs about ten minutes of pip
installs before any work can start. This image pays that once.

A pod built from it is ready in roughly a minute with **nothing to install**.

## Build it

GitHub Actions builds the image and pushes it to Docker Hub. Nothing large
crosses your connection and no local Docker is needed.

Add two repository secrets once — GitHub → **Settings** → **Secrets and
variables** → **Actions** → **New repository secret**:

| Name | Value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | a Docker Hub **access token**, not your password |

Create the token at Docker Hub → **Account settings** → **Personal access
tokens** → **Generate new token**, with *Read & Write* permission.

Then GitHub → **Actions** → **build pod image** → **Run workflow**. The first
build takes 20–40 minutes; later ones are faster because the layers are cached.
It reruns automatically whenever the image or the code baked into it changes.

The result is `YOURNAME/lookzi:latest`.

### Or build locally

Only if you would rather not use Actions.

```bash
cd docker
docker build -t YOURNAME/lookzi:latest .
docker push YOURNAME/lookzi:latest
```

## Use it on RunPod

When deploying a pod, choose **Custom template** (or edit the template) and set:

| | |
|---|---|
| Container image | `YOURNAME/lookzi:latest` |
| Container disk | 40 GB |
| Volume | 120 GB at `/workspace` |
| Expose HTTP ports | `7860` |

Exposing 7860 is worth doing: the URL becomes
`https://<POD_ID>-7860.proxy.runpod.net`, which survives a restart. A
`gradio.live` share link is regenerated on every launch and expires in 72 hours.

## What is inside

Two Python environments, deliberately separate:

| | |
|---|---|
| system python | try-on stack — the bundled diffusers `0.36.0.dev0` fork, transformers 4.56, hub 0.34.4 |
| `/opt/zimage-venv` | Z-Image-Turbo — diffusers from source, with the newer transformers and hub it requires |

They cannot share an interpreter. transformers 4.56 caps `huggingface_hub` below
1.0; diffusers-from-source wants 1.x. No single version satisfies both, and
several attempts to find one all failed — see the git history. Isolation ends it.

Also baked in: the DWPose ONNX models, the repo at `/opt/lookzi`, and the
environment variables that took a day to discover — xet and hf_transfer both
disabled, `expandable_segments` on, and the Z-Image cache pointed at local disk.

## What still happens per pod

Only what genuinely cannot be baked:

- **the 57.7 GB try-on model**, which lives on the volume and is reused across
  pods as long as the same volume is attached
- **12 GB of Z-Image weights**, unless built with `BAKE_ZIMAGE=1`
- a `git fetch` for the latest code

The entrypoint reports GPU health and whether the weights are present, then
leaves the container running for SSH.
