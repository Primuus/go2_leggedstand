"""Compile the shared smooth-ground profile into a MuJoCo height field."""
from pathlib import Path
import mujoco
import numpy as np
from go2_hindstand_lab.terrain_profile import profile, LENGTH, WIDTH, SLOPE_DEGREES, COLUMNS


def build_model(scene_path, mode='flat', level=0, column=5, seed=1):
    if mode not in ('flat', 'rolling') or not 0 <= level < len(SLOPE_DEGREES) or not 0 <= column < COLUMNS:
        raise ValueError('Invalid terrain mode, level or column')
    if mode == 'flat' or level == 0 or column < COLUMNS//2:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    _, z = profile(level, column, seed)
    low, high = float(z.min()), float(z.max())
    spec = mujoco.MjSpec.from_file(str(scene_path))
    spec.meshdir = str(Path(scene_path).parent/'obj')
    spec.add_hfield(name='rolling', nrow=2, ncol=len(z),
                    size=[LENGTH/2, WIDTH/2, high-low, .1],
                    userdata=np.tile((z-low)/(high-low), 2).tolist())
    floor = spec.geom('floor')
    floor.group = 5
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = 'rolling'
    floor.pos = [0., 0., low]
    return spec.compile()
