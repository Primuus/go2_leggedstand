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


def replace_event_inputs(value, insertion, old_count):
    """Retain contact/timing/support/height inputs; drop obsolete last-valid-leg semantics."""
    keep = 12 if old_count == 14 else 0
    return torch.cat((value[:, :insertion+keep], value.new_zeros(value.shape[0], EVENT_OBS_SIZE-keep),
                      value[:, insertion+old_count:]), dim=1)


def replace_clock_inputs(value, insertion):
    return replace_event_inputs(value, insertion, 2)


def migrate_event_checkpoint(checkpoint, model, old_count=2):
    old_actor, old_critic = 45+old_count, 86+old_count
    if checkpoint['model_state_dict']['actor.0.weight'].shape[1] != old_actor:
        raise ValueError(f'Migration requires a {old_actor}/{old_critic} checkpoint')
    assert model.actor[0].in_features == ACTOR_OBS_SIZE and model.critic[0].in_features == CRITIC_OBS_SIZE
    legacy = copy.deepcopy(model)
    legacy.actor[0] = torch.nn.Linear(old_actor, model.actor[0].out_features).to(model.actor[0].weight)
    legacy.critic[0] = torch.nn.Linear(old_critic, model.critic[0].out_features).to(model.critic[0].weight)
    legacy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    result = copy.deepcopy(checkpoint)
    insertions = {'actor.0.weight': 45, 'critic.0.weight': 48}
    for name, insertion in insertions.items():
        result['model_state_dict'][name] = replace_event_inputs(result['model_state_dict'][name], insertion, old_count)
    optimizer = result['optimizer_state_dict']
    ids = [p for group in optimizer['param_groups'] for p in group['params']]
    names = list(dict(model.named_parameters()))
    assert len(ids) == len(names)
    for name, param_id in zip(names, ids):
        if name in insertions:
            for field, value in optimizer['state'].get(param_id, {}).items():
                if torch.is_tensor(value) and value.ndim == 2:
                    optimizer['state'][param_id][field] = replace_event_inputs(value, insertions[name], old_count)
    model.load_state_dict(result['model_state_dict'], strict=True)
    keep = 12 if old_count == 14 else 0
    errors = {'old_actor_width': old_actor, 'new_actor_width': ACTOR_OBS_SIZE,
              'retained_state_columns': keep, 'dropped_state_columns': old_count-keep,
              'zero_initialized_state_columns': EVENT_OBS_SIZE-keep}
    with torch.no_grad():
        for name, width, insertion in (('actor', old_actor, 45), ('critic', old_critic, 48)):
            x = torch.randn(64, width, generator=torch.Generator().manual_seed(71)).to(model.actor[0].weight)
            full_legacy = getattr(legacy, name)(x)
            x[:, insertion+keep:insertion+old_count] = 0
            event = torch.ones(x.shape[0], EVENT_OBS_SIZE-keep, device=x.device, dtype=x.dtype)
            new_x = torch.cat((x[:, :insertion+keep], event, x[:, insertion+old_count:]), 1)
            before, after = getattr(legacy, name)(x), getattr(model, name)(new_x)
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
            errors[name+'_zero_removed_state_max_error'] = float((after-before).abs().max())
            errors[name+'_removed_state_synthetic_max_effect'] = float((after-full_legacy).abs().max())
    result['event_migration'] = errors
    return result
