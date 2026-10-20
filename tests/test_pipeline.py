from pathlib import Path

import h5py
import torch

from airport_cargo_mawm.data import collect_scripted_dataset
from airport_cargo_mawm.models import StateWorldModel
from airport_cargo_mawm.planning import CEMConfig, EnsembleCEMPlanner


def test_collection_schema(tmp_path: Path) -> None:
    output = tmp_path / "tiny.h5"
    summary = collect_scripted_dataset(output, episodes=2, seed=4, noise=0.0)
    assert summary["success_rate"] == 1.0
    with h5py.File(output, "r") as handle:
        assert handle["state"].shape[1] == 11
        assert handle["action"].shape[1] == 4
        assert handle["next_state"].shape == handle["state"].shape
        assert handle["truncated"].shape == handle["terminated"].shape
        assert set(handle["scene_id"][:]) == {0, 1}


def test_model_and_planner_shapes() -> None:
    model = StateWorldModel(hidden_dim=16)
    state = torch.zeros(3, 11)
    action = torch.zeros(3, 4)
    assert model(state, action).shape == (3, 11)
    planner = EnsembleCEMPlanner(
        [model],
        config=CEMConfig(horizon=2, num_samples=8, num_elites=2, iterations=1),
    )
    planned = planner.plan(state[0].numpy())
    assert planned.shape == (4,)
    assert ((planned >= -1.0) & (planned <= 1.0)).all()


def test_learned_planner_cannot_move_goal_or_time() -> None:
    class CheatingModel(torch.nn.Module):
        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            del action
            output = state.clone()
            output[:, 8:10] = output[:, 4:6]
            output[:, 10] = 0.0
            return output

    config = CEMConfig(
        horizon=2,
        num_samples=4,
        num_elites=2,
        iterations=1,
        success_bonus=0.0,
        terminal_goal_weight=1.0,
    )
    planner = EnsembleCEMPlanner([CheatingModel()], config=config)
    state = torch.tensor(
        [0.1, 0.4, 0.1, 0.6, 0.25, 0.5, 0.0, 0.0, 0.8, 0.5, 0.0]
    )
    actions = torch.zeros(config.num_samples, config.horizon, 4)
    costs = planner._cost(state, actions)
    assert torch.all(costs > 1.0), "planner accepted a model-imagined moving goal"
