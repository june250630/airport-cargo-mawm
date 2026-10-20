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
