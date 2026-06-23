#!/usr/bin/env bash
# Generate PickCube demos locally (motion-planning solver) + build the FPS zarr.
# Run from the repo root. No downloads.
# Extra args are forwarded to generate_demos.py, e.g. watch the solver live:
#   bash scripts/maniskill/prepare_pickcube_data.sh --vis --num-demos 5
set -e
export MUJOCO_GL=egl

python tools/maniskill/generate_demos.py --num-demos 200 "$@"

demos=data/maniskill_demos/PickCubeRL100-v1/motionplanning
python tools/maniskill/convert_states_to_zarr.py \
    --demo-h5 "$demos/trajectory.h5" --demo-json "$demos/trajectory.json" \
    --out data/maniskill_pickcube.zarr
