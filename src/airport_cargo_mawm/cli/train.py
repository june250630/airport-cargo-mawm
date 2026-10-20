from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from airport_cargo_mawm.data import load_transitions
from airport_cargo_mawm.models import StateWorldModel


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the state-space world model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/state_wm"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    arrays = load_transitions(args.data)
    scenes = np.unique(arrays["scene_id"])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(scenes)
    n_val = max(1, round(len(scenes) * args.val_fraction)) if len(scenes) > 1 else 0
    val_scenes = scenes[:n_val]
    val_mask = np.isin(arrays["scene_id"], val_scenes)
    train_mask = ~val_mask
    if not train_mask.any():
        raise ValueError("no training scenes remain after the validation split")

    def dataset(mask: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(arrays["state"][mask]).float(),
            torch.from_numpy(arrays["action"][mask]).float(),
            torch.from_numpy(arrays["next_state"][mask]).float(),
        )

    train_loader = DataLoader(
        dataset(train_mask), batch_size=args.batch_size, shuffle=True
    )
    val_loader = (
        DataLoader(dataset(val_mask), batch_size=args.batch_size)
        if val_mask.any()
        else None
    )
    model = StateWorldModel(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    history: list[dict[str, float | int]] = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))

    args.output.mkdir(parents=True, exist_ok=True)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for state, action, next_state in train_loader:
            state, action, next_state = (
                state.to(device),
                action.to(device),
                next_state.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(state, action), next_state)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(state)
            train_count += len(state)

        model.eval()
        val_total = 0.0
        val_count = 0
        if val_loader is not None:
            with torch.no_grad():
                for state, action, next_state in val_loader:
                    state, action, next_state = (
                        state.to(device),
                        action.to(device),
                        next_state.to(device),
                    )
                    loss = loss_fn(model(state, action), next_state)
                    val_total += float(loss) * len(state)
                    val_count += len(state)

        metrics = {
            "epoch": epoch,
            "train_loss": train_total / train_count,
            "val_loss": val_total / val_count if val_count else float("nan"),
        }
        history.append(metrics)
        print(json.dumps(metrics))
        checkpoint = {
            "epoch": epoch,
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "seed": args.seed,
            "data": str(args.data),
            "train_scenes": scenes[n_val:].tolist(),
            "val_scenes": val_scenes.tolist(),
        }
        torch.save(checkpoint, args.output / "last.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, args.output / f"epoch_{epoch:03d}.pt")

    (args.output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={args.output / 'last.pt'}; device={device}")


if __name__ == "__main__":
    main()
