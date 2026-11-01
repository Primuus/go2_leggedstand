"""Per-environment rolling-terrain curriculum; no simulator imports."""
import torch
from .terrain_profile import COLUMNS


class RollingCurriculum:
    def __init__(self, count, device, max_level=4, window=20):
        self.max_level, self.window = max_level, window
        self.success = torch.zeros(count, window, dtype=torch.bool, device=device)
        self.count = torch.zeros(count, dtype=torch.long, device=device)
        self.streak = torch.zeros_like(self.count)
        self.good_windows = torch.zeros_like(self.count)
        self.distance = torch.zeros(count, device=device)
        self.progress = torch.zeros_like(self.distance)
        self.moving_seconds = torch.zeros_like(self.distance)
        self.max_lateral = torch.zeros_like(self.distance)
        self.previous_x = torch.zeros_like(self.distance)
        self.freeze = False

    def advance(self, local_position, command, dt):
        moving = command > .05
        self.progress += (local_position[:, 0]-self.previous_x) * moving
        self.previous_x.copy_(local_position[:, 0])
        self.distance += command.clamp(min=0)*dt
        self.moving_seconds += moving*dt
        self.max_lateral = torch.maximum(self.max_lateral, local_position[:, 1].abs())

    def finish(self, ids, levels, columns, failed):
        """Ignore standing-only episodes. Two disjoint successful windows promote."""
        valid = ((self.moving_seconds[ids] >= 2.) & (self.distance[ids] >= .2)) | (failed & (self.moving_seconds[ids] > 0.))
        ratio = self.progress[ids]/self.distance[ids].clamp(min=.01)
        ok = ~failed & (ratio >= .8) & (self.max_lateral[ids] <= .5)
        ids, ok, failed = ids[valid], ok[valid], failed[valid]
        up = torch.zeros_like(levels, dtype=torch.bool)
        down = torch.zeros_like(up)
        if not self.freeze and len(ids):
            self.success[ids, self.count[ids] % self.window] = ok
            self.count[ids] += 1
            self.streak[ids] = torch.where(failed, self.streak[ids]+1, 0)
            full = self.count[ids] >= self.window
            rate = self.success[ids].float().mean(-1)
            boundary = self.count[ids] % self.window == 0
            self.good_windows[ids] = torch.where(boundary,
                torch.where(rate >= .85, self.good_windows[ids]+1, 0), self.good_windows[ids])
            up[ids] = full & (self.good_windows[ids] >= 2)
            down[ids] = ((full & (rate < .7)) | (self.streak[ids] >= 2)) & ~up[ids]
            up &= columns >= COLUMNS//2
            down &= columns >= COLUMNS//2
            up &= levels < self.max_level
            down &= levels > 0
            changed = up | down
            self.success[changed] = False
            self.count[changed] = 0
            self.streak[changed] = 0
            self.good_windows[changed] = 0
        return up, down

    def reset_episode(self, ids):
        for name in ('distance', 'progress', 'moving_seconds', 'max_lateral', 'previous_x'):
            getattr(self, name)[ids] = 0

    def state_dict(self):
        return {k: getattr(self, k).cpu().clone() for k in ('success', 'count', 'streak', 'good_windows')}

    def load_state_dict(self, data):
        for k in ('success', 'count', 'streak', 'good_windows'):
            target = getattr(self, k)
            if data[k].shape != target.shape:
                raise ValueError("Curriculum resume requires the same environment count")
            target.copy_(data[k].to(target.device))
