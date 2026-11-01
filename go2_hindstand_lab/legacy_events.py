"""Frozen 59-input observation adapter for replaying retained 4401/4901 policies."""
import torch


class LegacyEventObservation:
    def __init__(self, count, device, dt, cfg):
        self.dt, self.cfg = dt, cfg
        self.contact = torch.ones(count, 2, dtype=torch.bool, device=device)
        self.filtered = torch.ones_like(self.contact)
        self.air = torch.zeros(count, 2, device=device)
        self.single = torch.zeros_like(self.air)
        self.supported = torch.zeros_like(self.contact)
        self.peak = torch.zeros_like(self.air)
        self.last_valid = torch.zeros_like(self.contact)

    def reset(self, ids):
        self.contact[ids] = self.filtered[ids] = True
        for value in (self.air, self.single, self.supported, self.peak, self.last_valid):
            value[ids] = 0

    def update(self, forces, heights, enabled, body_weight):
        cfg = self.cfg
        contact = forces > 1.
        filtered = contact | self.contact
        air = ~filtered & enabled[:, None]
        landing = filtered & (self.air > 0) & enabled[:, None]
        other_loaded = forces.flip(1) > body_weight[:, None] * cfg.contact_support_weight_fraction
        beginning = air & (self.air == 0)
        self.supported.copy_(torch.where(beginning, other_loaded, self.supported))
        self.supported.logical_and_(other_loaded | ~air)
        self.peak.copy_(torch.where(air, torch.maximum(self.peak, heights.clamp_min(0)), self.peak))
        valid = (landing & self.supported & ~self.last_valid & (self.peak >= cfg.gait_valid_height)
                 & (self.air >= cfg.air_time_threshold - 1e-6)
                 & (self.air <= cfg.gait_max_air_time + 1e-6))
        completed = valid & (valid.sum(1, keepdim=True) == 1)
        self.last_valid.copy_(torch.where(completed.any(1, keepdim=True), completed, self.last_valid))
        self.last_valid.logical_and_(enabled[:, None])
        limit = cfg.gait_max_air_time + self.dt
        self.air.copy_(torch.where(air, (self.air + self.dt).clamp(max=limit), 0.))
        self.single.copy_(torch.where(air & other_loaded, (self.single + self.dt).clamp(max=limit), 0.))
        self.peak.copy_(torch.where(air, self.peak.clamp(max=2*cfg.target_foot_height), 0.))
        self.supported.logical_and_(air)
        self.contact.copy_(contact)
        self.filtered.copy_(filtered)

    def observation(self, moving):
        limit = self.cfg.gait_max_air_time + self.dt
        return torch.cat((self.contact, self.filtered, self.air / limit, self.single / limit,
                          self.supported, self.peak / (2*self.cfg.target_foot_height), self.last_valid), -1) * moving[:, None]
