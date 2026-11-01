"""Shared contact-driven gait state for Isaac Lab rewards and MuJoCo observations."""
import torch

EVENT_OBS_SIZE = 14
ACTOR_OBS_SIZE = 45 + EVENT_OBS_SIZE
CRITIC_OBS_SIZE = ACTOR_OBS_SIZE + 41


def gait_enabled(moving, upright_cosine, base_height, target_height):
    return moving & (upright_cosine > .7) & (torch.exp(-(base_height-target_height).abs()*5) > .7)


class GaitEvents:
    def __init__(self, count, device, dt, cfg):
        self.dt, self.cfg = dt, cfg
        self.contact = torch.ones(count, 2, dtype=torch.bool, device=device)
        self.filtered = torch.ones_like(self.contact)
        self.air = torch.zeros(count, 2, device=device)
        self.single = torch.zeros_like(self.air)
        self.supported = torch.zeros_like(self.contact)
        self.peak = torch.zeros_like(self.air)
        self.last_valid = torch.zeros_like(self.contact)
        self.air_score = torch.zeros(count, device=device)
        self.contact_score = torch.zeros_like(self.air_score)
        self.clearance_score = torch.zeros_like(self.air_score)
        self.completed = torch.zeros_like(self.contact)

    def reset(self, ids):
        self.contact[ids] = True
        self.filtered[ids] = True
        for value in (self.air, self.single, self.supported, self.peak, self.last_valid,
                      self.air_score, self.contact_score, self.clearance_score, self.completed):
            value[ids] = 0

    def update(self, forces, heights, enabled, body_weight):
        cfg = self.cfg
        contact = forces > 1.
        filtered = contact | self.contact
        air = ~filtered & enabled[:, None]
        landing = filtered & (self.air > 0) & enabled[:, None]
        other_loaded = forces.flip(1) > body_weight[:, None] * cfg.contact_support_weight_fraction
        beginning = air & (self.air == 0)
        ordered = ~self.last_valid
        self.supported.copy_(torch.where(beginning, other_loaded, self.supported))
        self.supported.logical_and_(other_loaded | ~air)
        self.peak.copy_(torch.where(air, torch.maximum(self.peak, heights.clamp_min(0)), self.peak))

        # Only complete, sufficiently high, supported and alternating steps advance history.
        valid = (landing & self.supported & ordered & (self.peak >= cfg.gait_valid_height)
                 & (self.air >= cfg.air_time_threshold - 1e-6)
                 & (self.air <= cfg.gait_max_air_time + 1e-6))
        self.completed.copy_(valid & (valid.sum(1, keepdim=True) == 1))
        value = ((self.air - cfg.air_time_threshold) / cfg.air_time_threshold).clamp(-1., 1.)
        value = torch.where(self.completed, value, value.clamp(max=0.))
        self.air_score.copy_((value * landing).sum(1).clamp(-1., 1.))
        self.last_valid.copy_(torch.where(self.completed.any(1, keepdim=True), self.completed, self.last_valid))
        self.last_valid.logical_and_(enabled[:, None])

        # Saturate only beyond the last reward transition; no hidden unbounded timer.
        limit = cfg.gait_max_air_time + self.dt
        self.air.copy_(torch.where(air, (self.air + self.dt).clamp(max=limit), 0.))
        single = air & other_loaded
        self.single.copy_(torch.where(single, (self.single + self.dt).clamp(max=limit), 0.))
        self.peak.copy_(torch.where(air, self.peak.clamp(max=2*cfg.target_foot_height), 0.))
        self.supported.logical_and_(air)

        progress = (self.single / cfg.air_time_threshold).clamp(0, 1)
        decay = ((cfg.gait_max_air_time - self.air) /
                 (cfg.gait_max_air_time - cfg.gait_decay_start)).clamp(0, 1)
        quality = single * (~self.last_valid) * progress * decay
        raised = heights > cfg.contact_min_clearance
        height_quality = (self.peak / cfg.gait_valid_height).clamp(0, 1)
        self.contact_score.copy_((quality * raised * height_quality).sum(1))
        height_score = (1 - (heights - cfg.target_foot_height).abs() / cfg.target_foot_height).clamp(0, 1)
        self.clearance_score.copy_((quality * height_score).sum(1))
        self.contact.copy_(contact)
        self.filtered.copy_(filtered)

    def observation(self, moving):
        """Seven left/right pairs: raw/filtered contacts, air/single timers, validity, peak, last leg."""
        limit = self.cfg.gait_max_air_time + self.dt
        obs = torch.cat((self.contact, self.filtered, self.air / limit, self.single / limit,
                         self.supported, self.peak / (2*self.cfg.target_foot_height), self.last_valid), -1)
        return obs * moving[:, None]
