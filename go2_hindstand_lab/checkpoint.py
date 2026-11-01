"""Migrate the 3300-era 45/86-input model and Adam state to explicit phase inputs."""
import copy
import torch
from .gait_events import EVENT_OBS_SIZE, ACTOR_OBS_SIZE, CRITIC_OBS_SIZE


def expand_input(value, insertion):
    zeros = value.new_zeros(value.shape[0], 2)
    return torch.cat((value[:, :insertion], zeros, value[:, insertion:]), dim=1)


def migrate_phase_checkpoint(checkpoint, model):
    legacy = copy.deepcopy(model)
    legacy.actor[0] = torch.nn.Linear(45, model.actor[0].out_features).to(model.actor[0].weight)
    legacy.critic[0] = torch.nn.Linear(86, model.critic[0].out_features).to(model.critic[0].weight)
    legacy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = copy.deepcopy(checkpoint)
    weights = result["model_state_dict"]
    expected = {"actor.0.weight": (45, 45), "critic.0.weight": (86, 48)}
    for name, (old_width, insertion) in expected.items():
        assert weights[name].shape[1] == old_width, (name, weights[name].shape)
        weights[name] = expand_input(weights[name], insertion)
    # Adam parameter IDs follow model.named_parameters(), as in the reference PPO.
    optimizer = result["optimizer_state_dict"]
    ids = [p for group in optimizer["param_groups"] for p in group["params"]]
    names = list(dict(model.named_parameters()))
    assert len(ids) == len(names)
    for name, param_id in zip(names, ids):
        if name in expected:
            old_width, insertion = expected[name]
            for field, value in optimizer["state"].get(param_id, {}).items():
                if torch.is_tensor(value) and value.ndim == 2:
                    assert value.shape[1] == old_width
                    optimizer["state"][param_id][field] = expand_input(value, insertion)
    model.load_state_dict(weights, strict=True)
    errors = {}
    with torch.no_grad():
        for name, width, insertion in (("actor", 45, 45), ("critic", 86, 48)):
            # Fixed private RNG: validation must not consume the training RNG stream.
            x = torch.randn(64, width, generator=torch.Generator().manual_seed(71)).to(model.actor[0].weight)
            expanded = torch.cat((x[:, :insertion], torch.ones_like(x[:, :2]), x[:, insertion:]), 1)
            before, after = getattr(legacy, name)(x), getattr(model, name)(expanded)
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
            errors[name + "_max_error"] = float((after-before).abs().max())
    result["phase_migration"] = errors
    return result


def replace_clock_inputs(value, insertion):
    """Keep physical inputs; replace two clock columns with zero event-state columns."""
    return torch.cat((value[:, :insertion], value.new_zeros(value.shape[0], EVENT_OBS_SIZE),
                      value[:, insertion+2:]), dim=1)


def migrate_event_checkpoint(checkpoint, model):
    if checkpoint['model_state_dict']['actor.0.weight'].shape[1] != 47:
        raise ValueError('Event migration requires a 47/88 phase checkpoint, e.g. model_4401.pt')
    assert model.actor[0].in_features == ACTOR_OBS_SIZE and model.critic[0].in_features == CRITIC_OBS_SIZE
    legacy = copy.deepcopy(model)
    legacy.actor[0] = torch.nn.Linear(47, model.actor[0].out_features).to(model.actor[0].weight)
    legacy.critic[0] = torch.nn.Linear(88, model.critic[0].out_features).to(model.critic[0].weight)
    legacy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    result = copy.deepcopy(checkpoint)
    insertions = {'actor.0.weight': 45, 'critic.0.weight': 48}
    for name, insertion in insertions.items():
        result['model_state_dict'][name] = replace_clock_inputs(result['model_state_dict'][name], insertion)
    optimizer = result['optimizer_state_dict']
    ids = [p for group in optimizer['param_groups'] for p in group['params']]
    names = list(dict(model.named_parameters()))
    assert len(ids) == len(names)
    for name, param_id in zip(names, ids):
        if name in insertions:
            for field, value in optimizer['state'].get(param_id, {}).items():
                if torch.is_tensor(value) and value.ndim == 2:
                    optimizer['state'][param_id][field] = replace_clock_inputs(value, insertions[name])
    model.load_state_dict(result['model_state_dict'], strict=True)
    errors = {}
    with torch.no_grad():
        for name, width, insertion in (('actor', 47, 45), ('critic', 88, 48)):
            x = torch.randn(64, width, generator=torch.Generator().manual_seed(71)).to(model.actor[0].weight)
            phase = torch.arange(64, device=x.device) * (2*torch.pi/64)
            x[:, insertion:insertion+2] = torch.stack((phase.sin(), phase.cos()), -1)
            full_legacy = getattr(legacy, name)(x)
            x[:, insertion:insertion+2] = 0
            event = torch.ones(x.shape[0], EVENT_OBS_SIZE, device=x.device, dtype=x.dtype)
            new_x = torch.cat((x[:, :insertion], event, x[:, insertion+2:]), 1)
            before, after = getattr(legacy, name)(x), getattr(model, name)(new_x)
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
            errors[name+'_zero_clock_max_error'] = float((after-before).abs().max())
            errors[name+'_removed_clock_max_effect'] = float((after-full_legacy).abs().max())
    result['event_migration'] = errors
    return result
