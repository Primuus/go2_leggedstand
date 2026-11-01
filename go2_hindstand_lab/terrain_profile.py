"""Shared rolling-ground geometry for Isaac Lab and MuJoCo (metres/radians)."""
from dataclasses import dataclass, asdict
import numpy as np
import torch

LENGTH, WIDTH, SPACING = 24., 6., .02
SLOPE_DEGREES = (0., 5., 7., 9., 12.)
COLUMNS = 10  # First half permanently flat; remaining columns have distinct rolling profiles.
START_PAD = 1.
FINISH_X = 10.
SIDE_LIMIT = 2.


def geometry_metadata():
    return dict(profile="alternating_cosine_lobes_v1", slope_degrees=list(SLOPE_DEGREES),
                length=LENGTH, width=WIDTH, spacing=SPACING, flat_columns=COLUMNS//2,
                columns=COLUMNS, start_pad=START_PAD, lobe_width_range=[.8, 1.2])


@dataclass(frozen=True)
class TerrainSettings:
    mode: str = "flat"
    seed: int = 1
    max_level: int = 4
    initial_level: int = 0
    min_level: int = 0
    keep_flat: bool = True

    def __post_init__(self):
        if self.mode not in ("flat", "rolling"):
            raise ValueError("Terrain must be flat or rolling")
        if not 0 <= self.min_level <= self.initial_level <= self.max_level < len(SLOPE_DEGREES):
            raise ValueError("Terrain levels must satisfy 0 <= min <= initial <= max <= 4")

    def to_dict(self):
        return asdict(self)


def initial_terrain_assignment(columns, settings):
    """Start rolling lanes at the requested level; optionally exclude every flat lane."""
    columns = columns.clone()
    if not settings.keep_flat:
        columns[:] = COLUMNS//2 + torch.arange(len(columns), device=columns.device) % (COLUMNS//2)
    levels = torch.full_like(columns, settings.initial_level)
    levels[columns < COLUMNS//2] = 0
    return levels, columns


def profile(level, column, seed=1):
    """C1 hills/valleys, constant across the lane; no steps or obstacles.

    Each cosine lobe starts and ends at zero with zero slope. Its maximum
    analytic slope is tan(level_angle). Linear mesh interpolation never exceeds it.
    """
    x = np.linspace(-LENGTH/2, LENGTH/2, round(LENGTH/SPACING)+1)
    z = np.zeros_like(x)
    if column < COLUMNS//2 or level == 0:
        return x, z
    rng = np.random.default_rng(seed + 101*column + 10007*level)
    start, sign = START_PAD, 1.
    # Keep a flat end strip before the mesh seam; the episode ends before this strip.
    while start < LENGTH/2 - 1.:
        width = float(rng.uniform(.8, 1.2))
        if start + width > LENGTH/2 - .5:
            break
        u = (np.abs(x)-start)/width
        inside = (u > 0) & (u < 1)
        amplitude = np.tan(np.deg2rad(SLOPE_DEGREES[level])) * width / (2*np.pi)
        z[inside] = sign * amplitude * (1-np.cos(2*np.pi*u[inside]))
        start += width
        sign *= -1
    return x, z


def mesh_arrays(level, column, seed=1):
    x, z = profile(level, column, seed)
    vertices = np.array([(a, y, h) for a, h in zip(x, z) for y in (-WIDTH/2, WIDTH/2)])
    faces = np.array([(2*i, 2*i+2, 2*i+1) for i in range(len(x)-1)] +
                     [(2*i+1, 2*i+2, 2*i+3) for i in range(len(x)-1)])
    return vertices, faces


class GroundProfiles:
    def __init__(self, seed=1, device="cpu"):
        self.values = torch.tensor(np.array([[profile(l, c, seed)[1] for c in range(COLUMNS)]
                                            for l in range(len(SLOPE_DEGREES))]),
                                   dtype=torch.float32, device=device)

    def height(self, x, levels, columns):
        """x [env, ...] is lane-local X; profiles are invariant along lane Y."""
        while levels.ndim < x.ndim:
            levels, columns = levels.unsqueeze(-1), columns.unsqueeze(-1)
        u = ((x + LENGTH/2)/SPACING).clamp(0, self.values.shape[-1]-1)
        i = u.long().clamp(max=self.values.shape[-1]-2)
        f = u-i
        return self.values[levels, columns, i]*(1-f) + self.values[levels, columns, i+1]*f

    def sole_ground(self, x, levels, columns, radius=.022):
        # Sphere footprint sampling at centre and +/-radius along X. Y is invariant.
        offsets = x.new_tensor([-radius, 0., radius])
        return self.height(x.unsqueeze(-1)+offsets, levels, columns).amax(-1)


def speed_limit(levels, columns):
    """Keep flat commands intact; learn rolling ground at lower speed first."""
    caps = levels.new_tensor([.5, .25, .30, .35, .40], dtype=torch.float32)
    return torch.where(columns < COLUMNS//2, .5, caps[levels])
