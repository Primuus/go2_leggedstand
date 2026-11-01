"""Shared policy clock for training and deployment; no sensor noise on the clock."""
import torch


def phase_observation(phase, moving):
    angle = 2 * torch.pi * phase
    return torch.stack((angle.sin(), angle.cos()), -1) * moving[..., None]
