"""Preflight checks on the actual loaded articulation, before PPO training."""
import torch
import xml.etree.ElementTree as ET
from isaaclab.utils.math import quat_apply
from .asset import asset_metadata, JOINT_NAMES, FOOT_NAMES, SOURCE
from .task import REF, SCALES, state


def validate(env):
    env.reset()
    s = state(env)
    robot = s.robot
    metadata = asset_metadata()
    actual_mass = robot.root_physx_view.get_masses().sum(dim=1)
    torch.testing.assert_close(actual_mass, torch.full_like(actual_mass, metadata["mass"]), rtol=1e-5, atol=1e-5)
    links = {link.get("name"): link for link in ET.parse(SOURCE).getroot().findall("link")}
    expected_inertia = []
    for name in robot.body_names:
        v = links[name].find("inertial/inertia").attrib
        expected_inertia.append([[float(v["ixx"]),float(v["ixy"]),float(v["ixz"])],
                                 [float(v["ixy"]),float(v["iyy"]),float(v["iyz"])],
                                 [float(v["ixz"]),float(v["iyz"]),float(v["izz"])]])
    actual_inertia = robot.data.default_inertia[0].reshape(-1, 3, 3)
    expected_eigenvalues = torch.linalg.eigvalsh(actual_inertia.new_tensor(expected_inertia))
    actual_eigenvalues = torch.linalg.eigvalsh(actual_inertia)
    torch.testing.assert_close(actual_eigenvalues, expected_eigenvalues, rtol=1e-4, atol=1e-7)
    for actual, value in ((robot.data.joint_damping, .1), (robot.data.joint_armature, .01),
                          (robot.data.joint_friction_coeff, .2)):
        torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=1e-6)
    root_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    torch.testing.assert_close(root_height, torch.full_like(root_height, .52), rtol=0, atol=1e-5)
    torch.testing.assert_close(s.dof_pos, s.default.expand_as(s.dof_pos), rtol=0, atol=1e-5)
    torch.testing.assert_close(s.projected_gravity, s.target_gravity.expand_as(s.projected_gravity), rtol=0, atol=1e-5)
    front = s.feet_pos[:, :, 2] - env.scene.env_origins[:, 2:3]
    torch.testing.assert_close(front, torch.full_like(front, .67), rtol=0, atol=1e-5)
    rear_ids = s.contact_foot_indices
    offset = robot.data.body_link_pos_w.new_tensor([-.002, 0., 0.]).expand(env.num_envs, 2, 3)
    rear_centers = robot.data.body_link_pos_w[:, rear_ids] + quat_apply(
        robot.data.body_link_quat_w[:, rear_ids].reshape(-1,4), offset.reshape(-1,3)).reshape(env.num_envs,2,3)
    rear_soles = rear_centers[..., 2] - .022 - env.scene.env_origins[:, 2:3]
    torch.testing.assert_close(rear_soles, torch.zeros_like(rear_soles), rtol=0, atol=1e-5)
    joint_limits = robot.data.joint_pos_limits[0, s.joint_ids]
    expected = joint_limits.new_tensor([[float(metadata["limits"][n][k]) for k in ("lower", "upper")] for n in JOINT_NAMES])
    torch.testing.assert_close(joint_limits, expected, rtol=0, atol=1e-5)
    assert env.action_manager.get_term("joints")._joint_names == JOINT_NAMES
    assert len(SCALES) == 23
    assert env.observation_manager.group_obs_dim == {"policy": (45,), "critic": (86,)}
    # Check the control action offset and explicit PD, independent of actor output.
    env.action_manager.process_action(torch.zeros(env.num_envs, 12, device=env.device))
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    torch.testing.assert_close(robot.data.applied_torque, torch.zeros_like(robot.data.applied_torque), rtol=0, atol=1e-3)
    s.refresh_tick = -1
    s.refresh()
    scores = {name: float(torch.as_tensor(getattr(s, "_reward_"+name)(), device=env.device).float().mean()) for name in SCALES}
    assert abs(scores["base_height"]-1) < 1e-4 and abs(scores["default_pos"]) < 1e-4
    assert abs(scores["handstand_orientation"]) < 1e-5 and scores["handstand_feet_height_exp"] > .9999
    report = {"mass_kg": float(actual_mass[0]), "joint_names": JOINT_NAMES, "feet": FOOT_NAMES,
              "native_joint_names": robot.joint_names, "native_body_names": robot.body_names,
              "initial_base_height_m": float(root_height[0]), "initial_front_foot_height_m": front[0].tolist(),
              "initial_hind_sole_height_m": rear_soles[0].tolist(),
              "observation_dims": [45, 86], "reward_count": 23, "initial_reward_scores": scores,
              "zero_action_initial_pd": "passed", "urdf_sha256": metadata["urdf_sha256"],
              "inertia_eigenvalue_max_error": float((actual_eigenvalues-expected_eigenvalues).abs().max()),
              "fixed_joint_parameters": "damping=.1, armature=.01, friction=.2 verified"}
    for _ in range(10):
        obs, rewards, _, _, _ = env.step(torch.zeros(env.num_envs, 12, device=env.device))
        assert torch.isfinite(obs["critic"]).all() and torch.isfinite(rewards).all()
    env.reset()
    s = state(env)
    torch.testing.assert_close(s.dof_pos, s.default.expand_as(s.dof_pos), rtol=0, atol=1e-5)
    report["zero_action_10_steps_and_reset"] = "finite; reset restored reference angles (not a balance test)"
    return report
