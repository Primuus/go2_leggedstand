"""Observable contact events and independently graded swing/placement quality."""
import torch

EVENT_OBS_SIZE = 22
ACTOR_OBS_SIZE = 45 + EVENT_OBS_SIZE
CRITIC_OBS_SIZE = ACTOR_OBS_SIZE + 41


def gait_enabled(moving, upright_cosine, base_height, target_height):
    return moving & (upright_cosine > .7) & (torch.exp(-(base_height-target_height).abs()*5) > .7)


def forward_foot_positions(centers, root_position, forward_world):
    """Horizontal local -Z projection; upright local X is vertical, not forward."""
    forward = forward_world[..., :2]
    forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1e-6)
    return ((centers[..., :2] - root_position[:, None, :2]) * forward[:, None]).sum(-1)


class GaitEvents:
    def __init__(self, count, device, dt, cfg):
        self.dt, self.cfg = dt, cfg
        self.contact = torch.ones(count, 2, dtype=torch.bool, device=device)
        self.filtered = torch.ones_like(self.contact)
        self.air = torch.zeros(count, 2, device=device)
        self.single = torch.zeros_like(self.air)
        self.support_ratio = torch.zeros_like(self.air)
        self.peak = torch.zeros_like(self.air)
        self.last_touch = torch.zeros_like(self.contact)
        self.landing_time = torch.zeros_like(self.air)
        self.start_span = torch.zeros_like(self.air)
        self.span = torch.zeros_like(self.air)
        self.landing_span = torch.zeros_like(self.air)
        # Outputs only: these do not influence subsequent state transitions.
        self.air_score = torch.zeros(count, device=device)
        self.contact_score = torch.zeros_like(self.air_score)
        self.clearance_score = torch.zeros_like(self.air_score)
        self.settled = torch.zeros_like(self.contact)
        self.reliable = torch.zeros_like(self.contact)
        self.alternating = torch.zeros_like(self.contact)
        self.completed = torch.zeros_like(self.contact)
        self.touchdown_span = torch.zeros_like(self.air)
        self.touchdown_air = torch.zeros_like(self.air)
        self.touchdown_support = torch.zeros_like(self.air)

    def reset(self, ids):
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                value[ids] = name in ('contact', 'filtered')

    def update(self, forces, heights, enabled, body_weight, foot_forward, command_vx):
        cfg = self.cfg
        active = enabled[:, None]
        contact = forces > 1.
        filtered = contact | self.contact
        airborne = ~filtered & active
        beginning = airborne & (self.air == 0)
        self.span.copy_(foot_forward - foot_forward.flip(1))
        self.start_span.copy_(torch.where(beginning, self.span, self.start_span))
        pending = contact & (self.air > 0) & active
        first_contact = pending & (self.landing_time == 0)
        self.landing_span.copy_(torch.where(first_contact, self.span, self.landing_span))
        self.landing_time.copy_(torch.where(pending, self.landing_time + self.dt, 0.))
        self.settled.copy_(pending & (self.landing_time >= cfg.gait_landing_confirm_s - 1e-6))
        self.peak.copy_(torch.where(airborne, torch.maximum(self.peak, heights.clamp_min(0)), self.peak).clamp(max=2*cfg.target_foot_height))

        # A brief force dip reduces average support, rather than invalidating the entire swing.
        load = (forces.flip(1) / (body_weight[:, None] * cfg.contact_support_weight_fraction)).clamp(0, 1)
        limit = cfg.gait_max_air_time + self.dt
        next_air = torch.where(airborne, (self.air + self.dt).clamp(max=limit), self.air)
        increment = next_air - self.air
        self.support_ratio.copy_((self.support_ratio*self.air + load*increment) / next_air.clamp_min(self.dt))
        self.air.copy_(next_air)
        self.single.copy_(torch.where(airborne & (load >= 1.), (self.single+self.dt).clamp(max=limit), 0.))
        self.reliable.copy_(self.settled & (self.air >= cfg.gait_reliable_air_s - 1e-6)
                            & (self.peak >= cfg.gait_reliable_height))
        unique = self.reliable & (self.reliable.sum(1, keepdim=True) == 1)
        ordered = ~self.last_touch
        self.alternating.copy_(unique & ordered)
        direction = command_vx.sign()[:, None]
        linear = (command_vx.abs() > cfg.gait_linear_threshold)[:, None]
        target = (command_vx.abs()*cfg.gait_stride_time).clamp(max=cfg.gait_max_stride)[:, None]
        target_safe = target.clamp_min(.01)
        signed_span = self.span * direction
        landing = self.landing_span * direction
        placement = (1 - (landing-target).abs()/target_safe).clamp(0, 1)
        # Diagnostic quality thresholds never control which leg must move next.
        self.completed.copy_(self.alternating & (self.air >= cfg.air_time_threshold-1e-6)
                             & (self.air <= cfg.gait_max_air_time+1e-6)
                             & (self.peak >= cfg.gait_valid_height)
                             & (self.support_ratio >= cfg.gait_valid_support_fraction)
                             & (~linear | (placement >= .5)))
        self.touchdown_span.copy_(landing*self.settled)
        self.touchdown_air.copy_(self.air*self.settled)
        self.touchdown_support.copy_(self.support_ratio*self.settled)
        # Duration can score positively even when the swing peak is below 4 cm.
        timing = ((self.air-cfg.air_time_threshold)/cfg.air_time_threshold).clamp(-1, 1)
        positive = timing.clamp_min(0) * self.support_ratio * self.alternating * (self.air <= cfg.gait_max_air_time+1e-6)
        self.air_score.copy_(((positive + timing.clamp(max=0))*self.settled).sum(1).clamp(-1, 1))

        progress = ((signed_span-self.start_span*direction) /
                    (target-self.start_span*direction).clamp_min(.01)).clamp(0, 1)
        progress *= ((2*target-signed_span)/target_safe).clamp(0, 1)
        height_score = (1-(heights-cfg.target_foot_height).abs()/cfg.target_foot_height).clamp(0, 1)
        # Equal continuous height / forward-progress scoring; no extra reward term or weight.
        swing_score = torch.where(linear, .5*(height_score+progress), height_score)
        decay = ((cfg.gait_max_air_time-self.air)/(cfg.gait_max_air_time-cfg.gait_decay_start)).clamp(0, 1)
        quality = airborne * ordered * load * self.support_ratio * decay
        self.clearance_score.copy_((quality*swing_score).sum(1).clamp(0, 1))
        support_progress = (self.air/cfg.air_time_threshold).clamp(0, 1)
        process = (quality*support_progress).sum(1)
        touchdown = (self.alternating*self.support_ratio*torch.where(linear, placement, 1.)
                     * (self.air <= cfg.gait_max_air_time+1e-6)).sum(1)
        self.contact_score.copy_((.5*process+.5*touchdown).clamp(0, 1))
        # Update leg history on reliable landings, independently of all quality thresholds.
        self.last_touch.copy_(torch.where(unique.any(1, keepdim=True), unique, self.last_touch) & active)
        retain = (self.air > 0) & ~self.settled & active
        for value in (self.air, self.single, self.support_ratio, self.peak,
                      self.landing_time, self.start_span, self.landing_span):
            value.mul_(retain)
        self.contact.copy_(contact)
        self.filtered.copy_(filtered)

    def observation(self, moving):
        """Eleven L/R pairs include every timer, quality accumulator and landing memory."""
        limit = self.cfg.gait_max_air_time + self.dt
        scale = self.cfg.gait_max_stride
        obs = torch.cat((self.contact, self.filtered, self.air/limit, self.single/limit,
                         self.support_ratio, self.peak/(2*self.cfg.target_foot_height), self.last_touch,
                         self.landing_time/self.cfg.gait_landing_confirm_s,
                         self.start_span/scale, self.span/scale, self.landing_span/scale), -1)
        return obs * moving[:, None]
