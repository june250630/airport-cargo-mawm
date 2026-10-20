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
    uncertainty_weight: float = 0.0
    collision_weight: float = 2.0


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
            for step in range(cfg.horizon):
                state = model(state, actions[:, step])
                state[:, :6] = state[:, :6].clamp(0.0, 1.0)
                state[:, 8:10] = state[:, 8:10].clamp(0.0, 1.0)
                cargo_distance = torch.linalg.vector_norm(
                    state[:, 4:6] - state[:, 8:10], dim=-1
                )
                tug_distance = torch.linalg.vector_norm(
                    state[:, 0:2] - state[:, 2:4], dim=-1
                )
                collision = torch.relu(0.06 - tug_distance)
                cumulative += cargo_distance + cfg.collision_weight * collision
            final_states.append(state)
            trajectory_costs.append(cumulative)

        model_cost = torch.stack(trajectory_costs).mean(dim=0)
        stacked = torch.stack(final_states)
        cargo_predictions = stacked[:, :, 4:6]
        disagreement = cargo_predictions.var(dim=0, unbiased=False).sum(dim=-1)
        return model_cost + cfg.uncertainty_weight * disagreement
