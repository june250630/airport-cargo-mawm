"""Reference policies used for solvability checks and data collection."""

from __future__ import annotations

import numpy as np

from airport_cargo_mawm.envs.airport_cargo import AirportCargoEnv


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def scripted_team_actions(
    env: AirportCargoEnv,
    *,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Stage both tugs, then push the ULD together using simulator state."""
    rng = rng or np.random.default_rng()
    goal_direction = _unit(env.goal_pos - env.cargo_pos)
    perpendicular = np.array([-goal_direction[1], goal_direction[0]], dtype=np.float32)
    distances = np.linalg.norm(env.tug_pos - env.cargo_pos, axis=1)
    ready = bool(np.all(distances < env.attach_radius * 0.82))

    actions: dict[str, np.ndarray] = {}
    for index, agent in enumerate(env.possible_agents):
        if ready:
            action = goal_direction
        else:
            side = -1.0 if index == 0 else 1.0
            staging = env.cargo_pos - 0.07 * goal_direction + side * 0.045 * perpendicular
            action = _unit(staging - env.tug_pos[index])
        if noise > 0:
            action = action + rng.normal(0.0, noise, size=2).astype(np.float32)
        actions[agent] = np.clip(action, -1.0, 1.0).astype(np.float32)
    return actions
