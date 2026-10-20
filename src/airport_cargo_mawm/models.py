"""Small state-space world model used by the first executable baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class StateWorldModel(nn.Module):
    """Predict a state delta from the current global state and joint action."""

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([state, action], dim=-1))
        return state + delta

    def config(self) -> dict[str, int]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }


def load_world_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[StateWorldModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = StateWorldModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
