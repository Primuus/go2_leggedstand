"""Contact-filtered rear-foot air-time reward, settled once at touchdown."""
import torch


class FeetAirTime:
    def __init__(self, count, device, dt, threshold, reward_cap, support_fraction):
        self.dt, self.threshold, self.reward_cap = dt, threshold, reward_cap
        self.support_fraction = support_fraction
        self.time = torch.zeros(count, 2, device=device)
        self.last_contact = torch.ones(count, 2, dtype=torch.bool, device=device)
        self.supported = torch.zeros_like(self.last_contact)
        self.score = torch.zeros(count, device=device)

    def reset(self, ids):
        self.time[ids] = 0
        self.last_contact[ids] = True
        self.supported[ids] = False
        self.score[ids] = 0

    def update(self, forces, enabled, body_weight):
        contact = forces > 1.
        filtered = contact | self.last_contact
        air = (~filtered) & enabled[:, None]
        landing = filtered & (self.time > 0) & enabled[:, None]
        other_loaded = forces.flip(1) > body_weight[:, None] * self.support_fraction
        # Every counted airborne frame must have a loaded opposite foot for positive credit.
        beginning = air & (self.time == 0)
        self.supported.copy_(torch.where(beginning, other_loaded, self.supported))
        self.supported.logical_and_(other_loaded | ~air)
        value = (self.time - self.threshold).clamp(max=self.reward_cap)
        value = torch.where(self.supported, value, value.clamp(max=0.))
        self.score.copy_((value * landing).sum(dim=1))
        self.time.copy_(torch.where(air, self.time + self.dt, 0.))
        self.supported.logical_and_(air)
        self.last_contact.copy_(torch.where(enabled[:, None], contact, True))
