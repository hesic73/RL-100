# RL-100 on NVIDIA Blackwell (RTX 50-series)

## Install

```bash
conda create -n rl100 python=3.10 -y
conda activate rl100
pip install --upgrade pip

# PyTorch first (needs the cu128 index)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements-blackwell.txt

# pytorch3d from in-repo source
PYTORCH3D_FORCE_NO_CUDA=1 pip install -e third_party/pytorch3d_simplified --no-build-isolation

# the rl_100 package (editable) so `import rl_100` works from any directory
pip install -e RL-100 --no-deps --no-build-isolation
```

## Environment variables

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # DMC headless GPU render
export SDL_VIDEODRIVER=dummy                 # PushT headless
```

## Run

ManiSkill PickCube (data generation, training, eval) is documented in `HANDOFF.md`.
Quick start from the repo root:

```bash
bash scripts/maniskill/prepare_pickcube_data.sh   # demos + data/maniskill_pickcube.zarr
bash scripts/maniskill/train_pickcube.sh          # offline BC
```
