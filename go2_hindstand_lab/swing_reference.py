"""Contact-timed targets; no external clock or new observation state."""
import math
import torch


def swing_reference(air, start_span, target_span, target_height, duration):
    u=(air/duration).clamp(0,1)
    smooth=u.square()*(3-2*u)
    height=target_height*torch.sin(math.pi*u).clamp_min(0)
    span=start_span+(target_span-start_span)*smooth
    return height,span


def swing_reference_score(air, heights, signed_span, signed_start, target, linear, cfg):
    h_ref,x_ref=swing_reference(air,signed_start,target,cfg.target_foot_height,cfg.gait_swing_target_s)
    h_score=torch.exp(-((heights-h_ref)/cfg.gait_swing_height_sigma).square())
    sigma=(target*cfg.gait_swing_span_sigma_fraction).clamp_min(cfg.gait_swing_min_span_sigma)
    x_score=torch.exp(-((signed_span-x_ref)/sigma).square())
    return h_score*torch.where(linear,x_score,torch.ones_like(x_score))
