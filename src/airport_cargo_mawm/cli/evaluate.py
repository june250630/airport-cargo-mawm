from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch

from airport_cargo_mawm.envs.airport_cargo import AirportCargoEnv
from airport_cargo_mawm.models import load_world_model
from airport_cargo_mawm.planning import CEMConfig, EnsembleCEMPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned-model CEM planning.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    models = [load_world_model(path, device=device)[0] for path in args.checkpoint]
    config = CEMConfig(
        horizon=args.horizon,
        num_samples=args.samples,
        num_elites=args.elites,
        iterations=args.iterations,
        uncertainty_weight=args.uncertainty_weight,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for episode in range(args.episodes):
        env = AirportCargoEnv(max_steps=80)
        env.reset(seed=args.seed + episode)
        planner = EnsembleCEMPlanner(
            models, config=config, device=device, seed=args.seed + episode
        )
        frames = [env.render()] if episode < args.videos else []
        total_reward = 0.0
        while env.agents:
            joint_action = planner.plan(env.global_state())
            actions = {"tug_0": joint_action[:2], "tug_1": joint_action[2:]}
            _, rewards, _, _, _ = env.step(actions)
            total_reward += rewards["tug_0"]
            if episode < args.videos:
                frames.append(env.render())
        result = {
            "episode": episode,
            "success": env.success,
            "steps": env.steps,
            "return": total_reward,
        }
        results.append(result)
        print(json.dumps(result))
        if frames:
            imageio.mimsave(
                args.output / f"episode_{episode:03d}.mp4",
                frames,
                fps=env.metadata["render_fps"],
                macro_block_size=1,
            )

    summary = {
        "episodes": args.episodes,
        "success_rate": sum(row["success"] for row in results) / args.episodes,
        "checkpoints": [str(path) for path in args.checkpoint],
        "uncertainty_weight": args.uncertainty_weight,
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"success_rate={summary['success_rate']:.1%}; output={args.output}")


if __name__ == "__main__":
    main()
