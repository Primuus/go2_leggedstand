"""Isaac Lab five-result API to the bundled reference PPO VecEnv API."""
import torch
from .task import state
from .gait_events import ACTOR_OBS_SIZE, CRITIC_OBS_SIZE


class ReferenceVecEnv:
    def __init__(self, env):
        self.env = env
        self.num_envs, self.device = env.num_envs, env.device
        self.num_obs, self.num_privileged_obs, self.num_actions = ACTOR_OBS_SIZE, CRITIC_OBS_SIZE, 12
        self.max_episode_length = env.max_episode_length
        self.diagnostic_log = state(env).diagnostics.summary()

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf[:] = value
        s = state(self.env)
        s.actor_cache, s.refresh_tick = None, -1
        self._observations(self.env.observation_manager.compute())

    def _observations(self, obs):
        self.obs_buf, self.privileged_obs_buf = obs["policy"], obs["critic"]
        if self.obs_buf.shape != (self.num_envs, self.num_obs) or self.privileged_obs_buf.shape != (self.num_envs, self.num_privileged_obs):
            raise RuntimeError("Reference policy/critic observation layout changed")
        if not torch.isfinite(self.privileged_obs_buf).all():
            raise RuntimeError("Non-finite observations")

    def reset(self):
        obs, _ = self.env.reset()
        self._observations(obs)
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError("Non-finite policy actions")
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        self._observations(obs)
        if not torch.isfinite(rewards).all():
            raise RuntimeError("Non-finite rewards")
        infos = {"time_outs": truncated}
        if extras.get("log"):
            infos["episode"] = dict(extras["log"])
        diagnostics = state(self.env).diagnostics
        if self.env.common_step_counter % 24 == 0:
            self.diagnostic_log = diagnostics.summary()
            diagnostics.clear_statistics()
        infos.setdefault("episode", {}).update({"Gait/" + k: v for k, v in self.diagnostic_log.items()})
        return self.obs_buf, self.privileged_obs_buf, rewards, terminated | truncated, infos
