"""Isaac Lab adapter for the shared smooth rolling-ground curriculum."""
import numpy as np
import torch
import trimesh
from isaaclab.managers import CurriculumTermCfg
from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass
from .terrain_profile import (TerrainSettings, GroundProfiles, mesh_arrays, LENGTH, WIDTH,
                              COLUMNS, SLOPE_DEGREES, geometry_metadata, initial_terrain_assignment)
from .terrain_curriculum import RollingCurriculum


def rolling_mesh(difficulty, cfg):
    level = min(int(difficulty*len(SLOPE_DEGREES)), len(SLOPE_DEGREES)-1)
    vertices, faces = mesh_arrays(level, cfg.column, cfg.profile_seed)
    vertices[:, :2] += np.array(cfg.size)/2
    return [trimesh.Trimesh(vertices=vertices, faces=faces, process=False)], np.array([cfg.size[0]/2, cfg.size[1]/2, 0.])


@configclass
class RollingTerrainCfg(SubTerrainBaseCfg):
    function = rolling_mesh
    column: int = 0
    profile_seed: int = 1


def update_levels(env, env_ids):
    if not hasattr(env, 'reference_state'):
        return 0.
    runtime = env.reference_state.ground
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    up, down = runtime.curriculum.finish(ids, runtime.levels, runtime.columns,
                                         env.termination_manager.terminated[ids])
    # Never ask TerrainImporter to overflow its last row (it randomly reassigns overflow).
    env.scene.terrain.update_env_origins(ids, up[ids], down[ids])
    runtime.curriculum.reset_episode(ids)
    return runtime.levels.float().mean()


@configclass
class CurriculumCfg:
    rolling_levels = CurriculumTermCfg(func=update_levels)


def configure_terrain(cfg, settings):
    cfg.terrain_settings = settings.to_dict()
    if settings.mode == 'flat':
        return
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(LENGTH, WIDTH), border_width=2., num_rows=len(SLOPE_DEGREES), num_cols=COLUMNS,
        curriculum=True, use_cache=False, seed=settings.seed,
        sub_terrains={f'lane_{c}': RollingTerrainCfg(proportion=1/COLUMNS, column=c,
                     profile_seed=settings.seed) for c in range(COLUMNS)})
    cfg.scene.terrain.max_init_terrain_level = settings.initial_level
    cfg.curriculum = CurriculumCfg()


class GroundRuntime:
    def __init__(self, env):
        self.env = env
        self.settings = TerrainSettings(**env.cfg.terrain_settings)
        self.profiles = GroundProfiles(self.settings.seed, env.device)
        self.curriculum = RollingCurriculum(env.num_envs, env.device, self.settings.max_level, min_level=self.settings.min_level)
        self.support_height = torch.zeros(env.num_envs, device=env.device)
        # Choose exact initial levels; optional all-rolling assignment excludes flat columns.
        if self.settings.mode == 'rolling':
            levels, columns = initial_terrain_assignment(self.columns, self.settings)
            self.levels.copy_(levels)
            self.columns.copy_(columns)
            env.scene.terrain.env_origins[:] = env.scene.terrain.terrain_origins[self.levels, self.columns]

    @property
    def levels(self):
        return self.env.scene.terrain.terrain_levels

    @property
    def columns(self):
        return self.env.scene.terrain.terrain_types

    def heights(self, centers):
        origins = self.env.scene.env_origins
        x = centers[..., 0]-origins[:, 0:1]
        if self.settings.mode == 'flat':
            return torch.zeros_like(x)+origins[:, 2:3]
        return self.profiles.sole_ground(x, self.levels, self.columns)+origins[:, 2:3]

    def update_support(self, heights, reset=None):
        target = heights.mean(-1)
        self.support_height.lerp_(target, .2)  # 100ms reference smoothing at50Hz.
        if reset is not None:
            self.support_height[reset] = target[reset]
        return self.support_height

    def state_dict(self):
        return dict(settings=self.settings.to_dict(), geometry=geometry_metadata(), levels=self.levels.cpu().clone(),
                    columns=self.columns.cpu().clone(), curriculum=self.curriculum.state_dict())

    def load_state_dict(self, data):
        if data['geometry'] != geometry_metadata():
            raise ValueError('Terrain geometry differs from checkpoint')
        if data['settings'] != self.settings.to_dict():
            raise ValueError('Terrain settings differ from checkpoint; use the recorded configuration')
        if data['levels'].shape != self.levels.shape:
            raise ValueError('Terrain resume requires the same number of environments')
        self.levels.copy_(data['levels'].to(self.levels.device))
        self.columns.copy_(data['columns'].to(self.columns.device))
        self.curriculum.load_state_dict(data['curriculum'])
        self.env.scene.terrain.env_origins[:] = self.env.scene.terrain.terrain_origins[self.levels, self.columns]

    def log(self):
        report = {'Terrain/level_mean': self.levels.float().mean(),
                  'Terrain/max_lateral_mean': self.curriculum.max_lateral.mean()}
        for l in range(len(SLOPE_DEGREES)):
            mask = (self.levels == l) & (self.columns >= COLUMNS//2)
            report[f'Terrain/L{l}_fraction'] = mask.float().mean()
            completed = mask & (self.curriculum.count >= self.curriculum.window)
            report[f'Terrain/L{l}_success_rate'] = (self.curriculum.success[completed].float().mean()
                if completed.any() else self.levels.new_tensor(0., dtype=torch.float32))
        return report
