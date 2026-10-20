"""Cross-Entropy Method planning with optional ensemble uncertainty cost."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from airport_cargo_mawm.models import StateWorldModel


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 8
    num_samples: int = 256
    num_elites: int = 32
    iterations: int = 4
    action_std: float = 0.8
    cost_mode: str = "task"
    uncertainty_weight: float = 0.0
    goal_weight: float = 1.0
    terminal_goal_weight: float = 5.0
    collision_weight: float = 5.0
    action_weight: float = 0.002
    attachment_weight: float = 1.0
    balance_weight: float = 0.1
    coupling_bonus: float = 0.20
    success_bonus: float = 20.0
    attach_radius: float = 0.15
    goal_radius: float = 0.07
    max_steps: int = 80


class EnsembleCEMPlanner:
    """Plan joint actions; ensemble disagreement acts as epistemic risk."""

    def __init__(
        self,
        models: list[StateWorldModel],
        *,
        config: CEMConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if not models:
            raise ValueError("at least one world model is required")
        self.models = [model.to(device).eval() for model in models]
        self.config = config or CEMConfig()
        if self.config.cost_mode not in {"task", "shaped"}:
            raise ValueError("cost_mode must be 'task' or 'shaped'")
        if self.config.num_elites > self.config.num_samples:
            raise ValueError("num_elites cannot exceed num_samples")
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def plan(self, state: np.ndarray) -> np.ndarray:
        cfg = self.config
        mean = torch.zeros((cfg.horizon, 4), device=self.device)
        std = torch.full_like(mean, cfg.action_std)
        initial = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        for _ in range(cfg.iterations):
            noise = torch.randn(
                (cfg.num_samples, cfg.horizon, 4),
                generator=self.generator,
                device=self.device,
            )
            candidates = torch.clamp(mean + std * noise, -1.0, 1.0)
            costs = self._cost(initial, candidates)
            elite_indices = torch.topk(costs, k=cfg.num_elites, largest=False).indices
            elites = candidates[elite_indices]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp_min(0.05)
        return mean[0].detach().cpu().numpy().astype(np.float32)

    def _cost(self, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        final_states = []
        trajectory_costs = []
        for model in self.models:
            state = initial.unsqueeze(0).expand(cfg.num_samples, -1).clone()
            cumulative = torch.zeros(cfg.num_samples, device=self.device)
            done = torch.zeros(
                cfg.num_samples, dtype=torch.bool, device=self.device
            )
            fixed_goal = initial[8:10].unsqueeze(0).expand(cfg.num_samples, -1)
            initial_time = initial[10]
            for step in range(cfg.horizon):
                action = actions[:, step]
                active = ~done
                predicted = model(state, action)
                predicted[:, :6] = predicted[:, :6].clamp(0.0, 1.0)
                # Goal and time are exogenous task variables, not dynamics that
                # the learned model is allowed to move to reduce imagined cost.
                predicted[:, 8:10] = fixed_goal
                predicted[:, 10] = torch.clamp(
                    initial_time + (step + 1) / cfg.max_steps, 0.0, 1.0
                )
                state = torch.where(active.unsqueeze(-1), predicted, state)
                cargo_distance = torch.linalg.vector_norm(
                    state[:, 4:6] - state[:, 8:10], dim=-1
                )
                tug_distance = torch.linalg.vector_norm(
                    state[:, 0:2] - state[:, 2:4], dim=-1
                )
                collision = tug_distance < 0.045
                success = cargo_distance <= cfg.goal_radius
                newly_successful = active & success
                timed_out = state[:, 10] >= 1.0
                done = done | success | timed_out

                step_cost = cfg.goal_weight * cargo_distance
                step_cost += cfg.collision_weight * (collision & active).float()
                step_cost += (
                    cfg.action_weight * action.square().sum(dim=-1) * active.float()
                )
                if cfg.cost_mode == "shaped":
                    tug_positions = state[:, 0:4].reshape(-1, 2, 2)
                    cargo = state[:, 4:6].unsqueeze(1)
                    distances = torch.linalg.vector_norm(
                        tug_positions - cargo, dim=-1
                    )
                    attachment_violation = torch.relu(
                        distances - cfg.attach_radius
                    ).sum(dim=-1)
                    balanced = torch.abs(distances[:, 0] - distances[:, 1])
                    action_pair = action.reshape(-1, 2, 2)
                    action_norms = torch.linalg.vector_norm(action_pair, dim=-1)
                    action_alignment = (
                        action_pair[:, 0] * action_pair[:, 1]
                    ).sum(dim=-1)
                    coupled_proxy = (
                        (distances <= cfg.attach_radius).all(dim=-1)
                        & (action_norms > 0.15).all(dim=-1)
                        & (action_alignment > 0.15)
                    )
                    step_cost += cfg.attachment_weight * attachment_violation
                    step_cost += cfg.balance_weight * balanced
                    step_cost -= cfg.coupling_bonus * (
                        coupled_proxy & active
                    ).float()
                step_cost -= cfg.success_bonus * newly_successful.float()
                cumulative += step_cost * active.float()
            final_states.append(state)
            final_distance = torch.linalg.vector_norm(
                state[:, 4:6] - state[:, 8:10], dim=-1
            )
            trajectory_costs.append(
                cumulative + cfg.terminal_goal_weight * final_distance
            )

        model_cost = torch.stack(trajectory_costs).mean(dim=0)
        stacked = torch.stack(final_states)
        cargo_predictions = stacked[:, :, 4:6]
        disagreement = cargo_predictions.var(dim=0, unbiased=False).sum(dim=-1)
        return model_cost + cfg.uncertainty_weight * disagreement
