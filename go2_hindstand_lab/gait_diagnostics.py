"""Read-only hind-foot timing diagnostics, inspired by robot_lab's contact/air timers.

No reward or observation depends on these statistics. Contact is vertical force >1 N.
Only complete walking air bouts are counted; reset/command transitions censor bouts.
"""
import torch


class GaitDiagnostics:
    def __init__(self, num_envs, device, dt):
        self.dt = dt
        self.contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.known = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.air_valid = torch.zeros_like(self.contact)
        self.air = torch.zeros(num_envs, 2, device=device)
        self.ground = torch.zeros_like(self.air)
        self.peak = torch.zeros_like(self.air)
        self.stats = torch.zeros(15, device=device, dtype=torch.float64)
        self.leg_count = torch.zeros(2, device=device, dtype=torch.float64)
        self.leg_time = torch.zeros_like(self.leg_count)
        self.leg_peak = torch.zeros_like(self.leg_count)

    def reset(self, ids):
        for value in (self.contact, self.known, self.air_valid, self.air, self.ground, self.peak):
            value[ids] = 0

    def clear_statistics(self):
        for value in (self.stats, self.leg_count, self.leg_time, self.leg_peak):
            value.zero_()

    def update(self, heights, forces, phase, moving):
        contact = forces > 1.
        landed = contact & ~self.contact & self.known[:, None] & self.air_valid & moving[:, None]
        self.leg_count += landed.sum(0)
        self.leg_time += (self.air * landed).sum(0)
        self.leg_peak += (self.peak * landed).sum(0)
        self.stats[10] += ((self.peak >= .04) & landed).sum()
        self.stats[11] += ((self.peak >= .06) & landed).sum()
        takeoff = ~contact & self.contact & self.known[:, None] & moving[:, None]
        self.air_valid.copy_((self.air_valid | takeoff) & ~contact & moving[:, None])
        self.air.copy_(torch.where(~contact, self.air + self.dt, 0.))
        self.ground.copy_(torch.where(contact, self.ground + self.dt, 0.))
        self.peak.copy_(torch.where(~contact, torch.maximum(self.peak, heights), 0.))
        single = (contact.sum(1) == 1) & moving
        expected_swing = torch.stack((phase >= .5, phase < .5), -1)
        matches = ((~contact) & expected_swing).any(1) & single
        middle = torch.sin(2 * torch.pi * phase).abs() >= 2/3
        expected_height = (heights * expected_swing).sum(1)
        mode_time = torch.where(contact, self.ground, self.air).min(1).values
        values = [moving.sum(), single.sum(), matches.sum(),
                  ((contact.sum(1) == 2) & moving).sum(),
                  ((contact.sum(1) == 0) & moving).sum(),
                  (mode_time * single).sum(), (forces.max(1).values * single).sum(),
                  (middle & moving).sum(),
                  ((expected_height >= .04) & middle & moving).sum(),
                  ((expected_height >= .06) & middle & moving).sum()]
        self.stats[:10] += torch.stack(values)
        self.stats[12] += ((heights >= .04).any(1) & moving).sum()
        self.stats[13] += ((heights >= .06).any(1) & moving).sum()
        self.stats[14] += (heights.clamp_min(0).max(1).values * moving).sum()
        self.contact.copy_(contact)
        self.known.fill_(True)

    def summary(self):
        s = self.stats
        walk, single, middle, bouts = s[0].clamp_min(1), s[1].clamp_min(1), s[7].clamp_min(1), self.leg_count.sum().clamp_min(1)
        result = {
            "walking_samples": s[0], "single_support_fraction": s[1]/walk,
            "phase_match_given_single": s[2]/single, "phase_matched_single_fraction": s[2]/walk,
            "double_support_fraction": s[3]/walk, "no_support_fraction": s[4]/walk,
            "single_support_elapsed_mean_s": s[5]/single,
            "single_support_force_mean_n": s[6]/single,
            "mid_swing_above_4cm_fraction": s[8]/middle, "mid_swing_above_6cm_fraction": s[9]/middle,
            "completed_air_above_4cm_fraction": s[10]/bouts, "completed_air_above_6cm_fraction": s[11]/bouts,
            "any_foot_above_4cm_fraction": s[12]/walk, "any_foot_above_6cm_fraction": s[13]/walk,
            "higher_foot_mean_m": s[14]/walk,
        }
        for i, leg in enumerate(("left", "right")):
            result[f"{leg}_completed_air_count"] = self.leg_count[i].clone()
            result[f"{leg}_completed_air_mean_s"] = self.leg_time[i]/self.leg_count[i].clamp_min(1)
            result[f"{leg}_completed_air_peak_mean_m"] = self.leg_peak[i]/self.leg_count[i].clamp_min(1)
        return {k: v.clone() for k, v in result.items()}
