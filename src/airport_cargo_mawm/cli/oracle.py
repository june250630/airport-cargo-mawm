from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from airport_cargo_mawm.envs.airport_cargo import AirportCargoEnv
from airport_cargo_mawm.oracle import OracleCEMConfig, OracleCEMPlanner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate privileged Oracle-dynamics CEM."
    )
    parser.add_argument("--cost", choices=["task", "shaped"], default="task")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="environment seed")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/oracle"))
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--elites", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    config = OracleCEMConfig(
        horizon=args.horizon,
        num_samples=args.samples,
        num_elites=args.elites,
        iterations=args.iterations,
        cost_mode=args.cost,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for episode in range(args.episodes):
        env = AirportCargoEnv(max_steps=80)
        env.reset(seed=args.seed + episode)
        planner = OracleCEMPlanner(
            config=config,
            device=device,
            seed=args.search_seed + episode,
        )
        frames = [env.render()] if episode < args.videos else []
        total_reward = 0.0
        coupled_steps = 0
        collisions = 0
        planning_seconds = 0.0
        first_coupling_step = None
        minimum_distance = float(np.linalg.norm(env.cargo_pos - env.goal_pos))
        while env.agents:
            start = time.perf_counter()
            joint_action = planner.plan(env.global_state())
            planning_seconds += time.perf_counter() - start
            actions = {"tug_0": joint_action[:2], "tug_1": joint_action[2:]}
            _, rewards, _, _, infos = env.step(actions)
            total_reward += rewards["tug_0"]
            coupled_steps += int(infos["tug_0"]["coupled"])
            collisions += int(infos["tug_0"]["collision"])
            if infos["tug_0"]["coupled"] and first_coupling_step is None:
                first_coupling_step = env.steps
            minimum_distance = min(
                minimum_distance, infos["tug_0"]["distance_to_goal"]
            )
            if episode < args.videos:
                frames.append(env.render())
        result = {
            "episode": episode,
            "environment_seed": args.seed + episode,
            "search_seed": args.search_seed + episode,
            "success": env.success,
            "steps": env.steps,
            "return": total_reward,
            "final_distance": float(np.linalg.norm(env.cargo_pos - env.goal_pos)),
            "minimum_distance": minimum_distance,
            "first_coupling_step": first_coupling_step,
            "coupled_steps": coupled_steps,
            "collisions": collisions,
            "planning_seconds": planning_seconds,
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
        "planner": "oracle_dynamics_cem",
        "privileged": True,
        "cost_mode": args.cost,
        "episodes": args.episodes,
        "success_rate": sum(row["success"] for row in results) / args.episodes,
        "mean_planning_seconds": sum(
            row["planning_seconds"] for row in results
        )
        / args.episodes,
        "mean_steps": sum(row["steps"] for row in results) / args.episodes,
        "mean_final_distance": sum(row["final_distance"] for row in results)
        / args.episodes,
        "total_collisions": sum(row["collisions"] for row in results),
        "cem": {
            "horizon": args.horizon,
            "samples": args.samples,
            "elites": args.elites,
            "iterations": args.iterations,
            "dynamics_evaluations_per_replan": (
                args.horizon * args.samples * args.iterations
            ),
        },
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"success_rate={summary['success_rate']:.1%}; output={args.output}")


if __name__ == "__main__":
    main()
