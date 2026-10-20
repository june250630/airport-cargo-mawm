import numpy as np
import torch

from airport_cargo_mawm.envs.airport_cargo import AirportCargoEnv
from airport_cargo_mawm.oracle import (
    OracleCEMConfig,
    OracleCEMPlanner,
    airport_cargo_true_step,
)


def _compare_one_step(env: AirportCargoEnv, joint_action: np.ndarray) -> None:
    initial = torch.from_numpy(env.global_state()).unsqueeze(0)
    action_tensor = torch.from_numpy(joint_action.astype(np.float32)).unsqueeze(0)
    predicted, info = airport_cargo_true_step(initial, action_tensor)
    actions = {"tug_0": joint_action[:2], "tug_1": joint_action[2:]}
    _, _, terminations, truncations, infos = env.step(actions)
    np.testing.assert_allclose(
        predicted[0].numpy(), env.global_state(), rtol=1e-6, atol=1e-6
    )
    assert bool(info.coupled[0]) == infos["tug_0"]["coupled"]
    assert bool(info.collision[0]) == infos["tug_0"]["collision"]
    assert bool(info.success[0]) == terminations["tug_0"]
    assert bool(info.truncated[0]) == truncations["tug_0"]


def test_true_dynamics_matches_random_environment_steps() -> None:
    rng = np.random.default_rng(21)
    env = AirportCargoEnv(max_steps=80)
    env.reset(seed=3)
    for _ in range(12):
        joint_action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        _compare_one_step(env, joint_action)


def test_true_dynamics_coupled_and_uncoupled_branches() -> None:
    coupled = AirportCargoEnv(max_steps=80)
    coupled.reset(
        options={
            "cargo_pos": [0.4, 0.5],
            "goal_pos": [0.8, 0.5],
            "tug_pos": [[0.30, 0.45], [0.30, 0.55]],
        }
    )
    _compare_one_step(coupled, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert coupled.last_coupled

    uncoupled = AirportCargoEnv(max_steps=80)
    uncoupled.reset(
        options={
            "cargo_pos": [0.5, 0.5],
            "goal_pos": [0.8, 0.5],
            "tug_pos": [[0.05, 0.05], [0.95, 0.95]],
        }
    )
    uncoupled.cargo_vel[:] = [0.2, -0.1]
    _compare_one_step(
        uncoupled, np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(uncoupled.cargo_vel, [0.1, -0.05], atol=1e-6)


def test_true_dynamics_matches_boundary_clipping() -> None:
    env = AirportCargoEnv(max_steps=80)
    env.reset(
        options={
            "cargo_pos": [0.95, 0.5],
            "goal_pos": [0.05, 0.5],
            "tug_pos": [[0.90, 0.45], [0.90, 0.55]],
        }
    )
    _compare_one_step(env, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_true_dynamics_matches_time_limit() -> None:
    env = AirportCargoEnv(max_steps=80)
    env.reset(seed=5)
    env.steps = 79
    _compare_one_step(env, np.zeros(4, dtype=np.float32))
    assert not env.agents
    assert not env.success


def test_oracle_planner_is_seed_deterministic() -> None:
    env = AirportCargoEnv()
    env.reset(seed=0)
    config = OracleCEMConfig(
        horizon=3, num_samples=16, num_elites=4, iterations=2, cost_mode="shaped"
    )
    first = OracleCEMPlanner(config=config, seed=123).plan(env.global_state())
    second = OracleCEMPlanner(config=config, seed=123).plan(env.global_state())
    np.testing.assert_allclose(first, second)


def test_oracle_task_cem_solves_fixed_episode() -> None:
    env = AirportCargoEnv(max_steps=80)
    env.reset(seed=0)
    planner = OracleCEMPlanner(
        config=OracleCEMConfig(
            horizon=20,
            num_samples=64,
            num_elites=8,
            iterations=3,
            cost_mode="task",
        ),
        seed=777,
    )
    collisions = 0
    while env.agents:
        joint_action = planner.plan(env.global_state())
        _, _, _, _, infos = env.step(
            {"tug_0": joint_action[:2], "tug_1": joint_action[2:]}
        )
        collisions += int(infos["tug_0"]["collision"])
    assert env.success
    assert collisions == 0
