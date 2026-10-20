"""Oracle-dynamics CEM for separating model error from search error.

The planner receives privileged global state and the exact environment
transition.  It is a diagnostic baseline, not a deployable policy and not a
mathematical upper bound: CEM can still fail to find a good action sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AirportCargoDynamicsConfig:
    dt: float = 0.1
    tug_speed: float = 0.55
    cargo_speed: float = 0.35
    attach_radius: float = 0.15
    goal_radius: float = 0.07
    max_steps: int = 80


@dataclass(frozen=True)
class OracleCEMConfig:
    horizon: int = 20
    num_samples: int = 1024
    num_elites: int = 64
    iterations: int = 5
    action_std: float = 0.9
    cost_mode: str = "task"
    goal_weight: float = 1.0
    terminal_goal_weight: float = 5.0
    collision_weight: float = 5.0
    action_weight: float = 0.002
    attachment_weight: float = 1.0
    balance_weight: float = 0.1
    coupling_bonus: float = 0.20
    success_bonus: float = 20.0


@dataclass(frozen=True)
class OracleStepInfo:
    coupled: torch.Tensor
    collision: torch.Tensor
    success: torch.Tensor
    truncated: torch.Tensor


def airport_cargo_true_step(
    state: torch.Tensor,
    joint_action: torch.Tensor,
    *,
    config: AirportCargoDynamicsConfig | None = None,
) -> tuple[torch.Tensor, OracleStepInfo]:
    """Vectorized Torch copy of one ``AirportCargoEnv.step`` transition.

    ``state`` has shape ``(..., 11)`` and ``joint_action`` has shape
    ``(..., 4)``.  Equivalence with the NumPy environment is covered by tests.
    """
    cfg = config or AirportCargoDynamicsConfig()
    if state.shape[-1] != 11:
        raise ValueError(f"expected state dimension 11, got {state.shape[-1]}")
    if joint_action.shape[-1] != 4:
        raise ValueError(
            f"expected joint action dimension 4, got {joint_action.shape[-1]}"
        )
    if state.shape[:-1] != joint_action.shape[:-1]:
        raise ValueError("state and joint_action batch dimensions must match")

    action = joint_action.clamp(-1.0, 1.0).reshape(*joint_action.shape[:-1], 2, 2)
    tugs = state[..., 0:4].reshape(*state.shape[:-1], 2, 2)
    cargo = state[..., 4:6]
    cargo_velocity = state[..., 6:8]
    goal = state[..., 8:10]
    time_fraction = state[..., 10:11]

    proposed_tugs = (tugs + cfg.tug_speed * cfg.dt * action).clamp(0.02, 0.98)
    distances = torch.linalg.vector_norm(proposed_tugs - cargo.unsqueeze(-2), dim=-1)
    action_norms = torch.linalg.vector_norm(action, dim=-1)
    alignment = (action[..., 0, :] * action[..., 1, :]).sum(dim=-1)
    coupled = (
        (distances <= cfg.attach_radius).all(dim=-1)
        & (action_norms > 0.15).all(dim=-1)
        & (alignment > 0.15)
    )

    mean_action = action.mean(dim=-2)
    delta = cfg.cargo_speed * cfg.dt * mean_action
    moved_cargo = (cargo + delta).clamp(0.04, 0.96)
    moved_tugs = (tugs + delta.unsqueeze(-2)).clamp(0.02, 0.98)
    next_cargo = torch.where(coupled.unsqueeze(-1), moved_cargo, cargo)
    next_tugs = torch.where(coupled.unsqueeze(-1).unsqueeze(-1), moved_tugs, proposed_tugs)
    next_velocity = torch.where(
        coupled.unsqueeze(-1), delta / cfg.dt, cargo_velocity * 0.5
    )
    next_time = (time_fraction + 1.0 / cfg.max_steps).clamp(0.0, 1.0)

    next_state = torch.cat(
        [
            next_tugs.flatten(start_dim=-2),
            next_cargo,
            next_velocity,
            goal,
            next_time,
        ],
        dim=-1,
    )
    tug_separation = torch.linalg.vector_norm(
        next_tugs[..., 0, :] - next_tugs[..., 1, :], dim=-1
    )
    collision = tug_separation < 0.045
    success = torch.linalg.vector_norm(next_cargo - goal, dim=-1) <= cfg.goal_radius
    truncated = (next_time.squeeze(-1) >= 1.0) & ~success
    return next_state, OracleStepInfo(coupled, collision, success, truncated)


class OracleCEMPlanner:
    """Centralized CEM using the exact airport-cargo transition."""

    def __init__(
        self,
        *,
        config: OracleCEMConfig | None = None,
        dynamics: AirportCargoDynamicsConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config or OracleCEMConfig()
        self.dynamics = dynamics or AirportCargoDynamicsConfig()
        if self.config.cost_mode not in {"task", "shaped"}:
            raise ValueError("cost_mode must be 'task' or 'shaped'")
        if self.config.num_elites > self.config.num_samples:
            raise ValueError("num_elites cannot exceed num_samples")
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._warm_mean: torch.Tensor | None = None

    def reset(self) -> None:
        self._warm_mean = None

    @torch.no_grad()
    def plan(self, state: np.ndarray) -> np.ndarray:
        cfg = self.config
        if self._warm_mean is None:
            mean = torch.zeros((cfg.horizon, 4), device=self.device)
        else:
            mean = self._warm_mean.clone()
        std = torch.full_like(mean, cfg.action_std)
        initial = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        for _ in range(cfg.iterations):
            noise = torch.randn(
                (cfg.num_samples, cfg.horizon, 4),
                generator=self.generator,
                device=self.device,
            )
            candidates = (mean + std * noise).clamp(-1.0, 1.0)
            candidates[0] = mean.clamp(-1.0, 1.0)
            costs = self._cost(initial, candidates)
            elite_indices = torch.topk(costs, k=cfg.num_elites, largest=False).indices
            elites = candidates[elite_indices]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp_min(0.05)

        self._warm_mean = torch.cat([mean[1:], torch.zeros_like(mean[:1])], dim=0)
        return mean[0].cpu().numpy().astype(np.float32)

    def _cost(self, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        state = initial.unsqueeze(0).expand(cfg.num_samples, -1).clone()
        cumulative = torch.zeros(cfg.num_samples, device=self.device)
        done = torch.zeros(cfg.num_samples, dtype=torch.bool, device=self.device)

        for step in range(cfg.horizon):
            action = actions[:, step]
            predicted, info = airport_cargo_true_step(
                state, action, config=self.dynamics
            )
            active = ~done
            state = torch.where(active.unsqueeze(-1), predicted, state)
            newly_successful = active & info.success
            done = done | info.success | info.truncated

            cargo_goal_distance = torch.linalg.vector_norm(
                state[:, 4:6] - state[:, 8:10], dim=-1
            )
            step_cost = cfg.goal_weight * cargo_goal_distance
            step_cost += cfg.collision_weight * (info.collision & active).float()
            step_cost += cfg.action_weight * action.square().sum(dim=-1) * active.float()

            if cfg.cost_mode == "shaped":
                tug_positions = state[:, 0:4].reshape(-1, 2, 2)
                cargo = state[:, 4:6].unsqueeze(1)
                distances = torch.linalg.vector_norm(tug_positions - cargo, dim=-1)
                attachment_violation = torch.relu(
                    distances - self.dynamics.attach_radius
                ).sum(dim=-1)
                action_pair = action.reshape(-1, 2, 2)
                action_norms = torch.linalg.vector_norm(action_pair, dim=-1)
                action_alignment = (
                    action_pair[:, 0] * action_pair[:, 1]
                ).sum(dim=-1)
                coupled_proxy = (
                    (distances <= self.dynamics.attach_radius).all(dim=-1)
                    & (action_norms > 0.15).all(dim=-1)
                    & (action_alignment > 0.15)
                )
                step_cost += cfg.attachment_weight * attachment_violation
                step_cost += cfg.balance_weight * torch.abs(
                    distances[:, 0] - distances[:, 1]
                )
                step_cost -= cfg.coupling_bonus * (
                    coupled_proxy & active
                ).float()

            step_cost -= cfg.success_bonus * newly_successful.float()
            cumulative += step_cost * active.float()

        final_distance = torch.linalg.vector_norm(
            state[:, 4:6] - state[:, 8:10], dim=-1
        )
        return cumulative + cfg.terminal_goal_weight * final_distance
