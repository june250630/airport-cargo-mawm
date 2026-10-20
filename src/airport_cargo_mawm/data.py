"""HDF5 trajectory collection and loading utilities."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from airport_cargo_mawm.envs.airport_cargo import AirportCargoEnv
from airport_cargo_mawm.policies import scripted_team_actions


def collect_scripted_dataset(
    output: str | Path,
    *,
    episodes: int = 100,
    seed: int = 0,
    max_steps: int = 80,
    noise: float = 0.03,
) -> dict[str, float | int | str]:
    """Collect scene-aligned transitions from the scripted oracle team."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "state": [],
        "action": [],
        "next_state": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "scene_id": [],
        "step": [],
    }
    successes = 0

    for episode in range(episodes):
        env = AirportCargoEnv(max_steps=max_steps, observation_mode="state")
        env.reset(seed=seed + episode)
        while env.agents:
            state = env.global_state()
            actions = scripted_team_actions(env, noise=noise, rng=rng)
            joint_action = np.concatenate(
                [actions[agent] for agent in env.possible_agents]
            ).astype(np.float32)
            _, rewards, terminations, truncations, _ = env.step(actions)
            rows["state"].append(state)
            rows["action"].append(joint_action)
            rows["next_state"].append(env.global_state())
            rows["reward"].append(float(rewards["tug_0"]))
            rows["terminated"].append(bool(terminations["tug_0"]))
            rows["truncated"].append(bool(truncations["tug_0"]))
            rows["scene_id"].append(episode)
            rows["step"].append(env.steps - 1)
        successes += int(env.success)
        env.close()

    with h5py.File(output, "w") as handle:
        handle.create_dataset("state", data=np.asarray(rows["state"], dtype=np.float32))
        handle.create_dataset("action", data=np.asarray(rows["action"], dtype=np.float32))
        handle.create_dataset(
            "next_state", data=np.asarray(rows["next_state"], dtype=np.float32)
        )
        handle.create_dataset("reward", data=np.asarray(rows["reward"], dtype=np.float32))
        handle.create_dataset(
            "terminated", data=np.asarray(rows["terminated"], dtype=np.bool_)
        )
        handle.create_dataset(
            "truncated", data=np.asarray(rows["truncated"], dtype=np.bool_)
        )
        handle.create_dataset(
            "scene_id", data=np.asarray(rows["scene_id"], dtype=np.int64)
        )
        handle.create_dataset("step", data=np.asarray(rows["step"], dtype=np.int32))
        handle.attrs["seed"] = seed
        handle.attrs["episodes"] = episodes
        handle.attrs["max_steps"] = max_steps
        handle.attrs["success_rate"] = successes / episodes
        handle.attrs["state_layout"] = (
            "tug0_xy,tug1_xy,cargo_xy,cargo_velocity_xy,goal_xy,time_fraction"
        )

    return {
        "path": str(output),
        "episodes": episodes,
        "transitions": len(rows["state"]),
        "success_rate": successes / episodes,
    }


def load_transitions(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {key: handle[key][:] for key in handle.keys()}
