"""Migrate the 3300-era 45/86-input model and Adam state to explicit phase inputs."""
import copy
import torch


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
