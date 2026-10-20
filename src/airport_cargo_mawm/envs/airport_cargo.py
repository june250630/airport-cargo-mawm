"""A small cooperative airport-cargo environment.

Two ground tugs must approach a unit load device (ULD) and push it toward an
aircraft stand. The ULD only moves when both tugs are close and their actions
point in approximately the same direction. This executable specification is
used before the project moves to a VMAS scenario.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


class AirportCargoEnv(ParallelEnv):
    """PettingZoo parallel environment for two-tug cooperative transport."""

    metadata = {
        "name": "airport_cargo_v0",
        "render_modes": ["rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        *,
        max_steps: int = 80,
        observation_mode: str = "state",
        image_size: int = 128,
        dt: float = 0.1,
        tug_speed: float = 0.55,
        cargo_speed: float = 0.35,
        attach_radius: float = 0.15,
        goal_radius: float = 0.07,
        render_mode: str | None = "rgb_array",
    ) -> None:
        if observation_mode not in {"state", "pixels"}:
            raise ValueError("observation_mode must be 'state' or 'pixels'")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("only render_mode='rgb_array' is supported")

        self.max_steps = int(max_steps)
        self.observation_mode = observation_mode
        self.image_size = int(image_size)
        self.dt = float(dt)
        self.tug_speed = float(tug_speed)
        self.cargo_speed = float(cargo_speed)
        self.attach_radius = float(attach_radius)
        self.goal_radius = float(goal_radius)
        self.render_mode = render_mode

        self.possible_agents = ["tug_0", "tug_1"]
        self.agents: list[str] = []
        self._action_spaces = {
            agent: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }
        if observation_mode == "state":
            obs_space = spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float32)
        else:
            obs_space = spaces.Box(
                0, 255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
            )
        self._observation_spaces = {
            agent: obs_space for agent in self.possible_agents
        }

        self.np_random = np.random.default_rng()
        self.tug_pos = np.zeros((2, 2), dtype=np.float32)
        self.cargo_pos = np.zeros(2, dtype=np.float32)
        self.cargo_vel = np.zeros(2, dtype=np.float32)
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.success = False
        self.last_coupled = False

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        jitter = self.np_random.uniform(-0.025, 0.025, size=2).astype(np.float32)
        self.cargo_pos = np.asarray(
            options.get("cargo_pos", np.array([0.28, 0.50], dtype=np.float32) + jitter),
            dtype=np.float32,
        )
        self.goal_pos = np.asarray(
            options.get("goal_pos", [0.82, 0.50]), dtype=np.float32
        )
        self.tug_pos = np.asarray(
            options.get(
                "tug_pos",
                [
                    self.cargo_pos + np.array([-0.17, -0.08], dtype=np.float32),
                    self.cargo_pos + np.array([-0.17, 0.08], dtype=np.float32),
                ],
            ),
            dtype=np.float32,
        )
        self.tug_pos = np.clip(self.tug_pos, 0.02, 0.98)
        self.cargo_vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.success = False
        self.last_coupled = False
        self.agents = self.possible_agents.copy()
        observations = self._observations()
        infos = {agent: self._info() for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("step() called after episode end; call reset()")
        missing = set(self.agents) - set(actions)
        if missing:
            raise KeyError(f"missing actions for {sorted(missing)}")

        active_agents = self.agents.copy()
        joint_action = np.stack(
            [np.asarray(actions[a], dtype=np.float32) for a in active_agents]
        )
        joint_action = np.clip(joint_action, -1.0, 1.0)
        old_tug_pos = self.tug_pos.copy()
        proposed_tug_pos = np.clip(
            old_tug_pos + self.tug_speed * self.dt * joint_action, 0.02, 0.98
        )

        distances = np.linalg.norm(proposed_tug_pos - self.cargo_pos, axis=1)
        norms = np.linalg.norm(joint_action, axis=1)
        alignment = float(np.dot(joint_action[0], joint_action[1]))
        both_active = bool(np.all(norms > 0.15))
        coupled = bool(
            np.all(distances <= self.attach_radius)
            and both_active
            and alignment > 0.15
        )

        if coupled:
            mean_action = joint_action.mean(axis=0)
            delta = self.cargo_speed * self.dt * mean_action
            self.cargo_pos = np.clip(self.cargo_pos + delta, 0.04, 0.96)
            self.cargo_vel = delta / self.dt
            self.tug_pos = np.clip(old_tug_pos + delta, 0.02, 0.98)
        else:
            self.tug_pos = proposed_tug_pos
            self.cargo_vel *= 0.5

        self.last_coupled = coupled
        self.steps += 1
        distance_to_goal = float(np.linalg.norm(self.cargo_pos - self.goal_pos))
        tug_distance = float(np.linalg.norm(self.tug_pos[0] - self.tug_pos[1]))
        collision = tug_distance < 0.045
        self.success = distance_to_goal <= self.goal_radius
        truncated = self.steps >= self.max_steps and not self.success

        reward = -distance_to_goal - 0.01
        if coupled:
            reward += 0.05
        if collision:
            reward -= 0.5
        if self.success:
            reward += 10.0

        observations = self._observations()
        rewards = {agent: float(reward) for agent in active_agents}
        terminations = {agent: self.success for agent in active_agents}
        truncations = {agent: truncated for agent in active_agents}
        infos = {agent: self._info(collision=collision) for agent in active_agents}
        if self.success or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def global_state(self) -> np.ndarray:
        """Return a normalized Markov state in a fixed, documented order."""
        return np.concatenate(
            [
                self.tug_pos.reshape(-1),
                self.cargo_pos,
                self.cargo_vel,
                self.goal_pos,
                np.array([self.steps / self.max_steps], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _state_observation(self, agent_index: int) -> np.ndarray:
        partner = 1 - agent_index
        return np.concatenate(
            [
                self.tug_pos[agent_index],
                self.tug_pos[partner],
                self.cargo_pos,
                self.cargo_vel,
                self.goal_pos,
                np.array([self.steps / self.max_steps], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _observations(self) -> dict[str, np.ndarray]:
        if self.observation_mode == "state":
            return {
                agent: self._state_observation(i)
                for i, agent in enumerate(self.agents)
            }
        return {
            agent: self._render_frame(viewer_agent=i)
            for i, agent in enumerate(self.agents)
        }

    def _info(self, *, collision: bool = False) -> dict[str, Any]:
        return {
            "success": self.success,
            "coupled": self.last_coupled,
            "collision": bool(collision),
            "distance_to_goal": float(np.linalg.norm(self.cargo_pos - self.goal_pos)),
            "step": self.steps,
        }

    def render(self) -> np.ndarray:
        return self._render_frame(viewer_agent=None)

    def _render_frame(self, viewer_agent: int | None) -> np.ndarray:
        size = self.image_size
        frame = np.full((size, size, 3), 242, dtype=np.uint8)
        frame[:3] = 70
        frame[-3:] = 70
        frame[:, :3] = 70
        frame[:, -3:] = 70

        def pixel(point: np.ndarray) -> tuple[int, int]:
            x = int(np.clip(point[0], 0, 1) * (size - 1))
            y = int((1.0 - np.clip(point[1], 0, 1)) * (size - 1))
            return x, y

        def disk(point: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
            x, y = pixel(point)
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            frame[mask] = color

        def square(point: np.ndarray, half: int, color: tuple[int, int, int]) -> None:
            x, y = pixel(point)
            frame[
                max(0, y - half) : min(size, y + half + 1),
                max(0, x - half) : min(size, x + half + 1),
            ] = color

        disk(self.goal_pos, max(7, size // 15), (126, 210, 120))
        disk(self.goal_pos, max(3, size // 30), (242, 242, 242))
        square(self.cargo_pos, max(4, size // 25), (85, 85, 90))
        for idx, position in enumerate(self.tug_pos):
            if viewer_agent is None:
                color = (45, 105, 210) if idx == 0 else (235, 135, 35)
            else:
                color = (45, 105, 210) if idx == viewer_agent else (235, 135, 35)
            disk(position, max(4, size // 28), color)
        return frame

    def close(self) -> None:
        return None
