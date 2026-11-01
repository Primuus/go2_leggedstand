"""Keep rolling-terrain progress together with model and optimizer checkpoints."""
from rsl_rl.runners import OnPolicyRunner
from .task import state


class TrainingRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        data = dict(infos or {})
        ground = state(self.env.env).ground
        if ground.settings.mode == 'rolling':
            data['terrain_curriculum'] = ground.state_dict()
        super().save(path, infos=data)

    def load(self, path, load_optimizer=True):
        infos = super().load(path, load_optimizer)
        if infos and 'terrain_curriculum' in infos:
            ground = state(self.env.env).ground
            ground.load_state_dict(infos['terrain_curriculum'])
            # Clear unfinished episode progress but retain completed curriculum windows.
            ground.curriculum.freeze = True
            self.env.reset()
            ground.curriculum.freeze = False
        return infos
