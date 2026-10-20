from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from airport_cargo_mawm.envs.airport_cargo import AirportCargoEnv
from airport_cargo_mawm.policies import scripted_team_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scripted solvability check.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--videos", type=int, default=2)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for episode in range(args.episodes):
        env = AirportCargoEnv(max_steps=80)
        env.reset(seed=args.seed + episode)
        frames = [env.render()] if episode < args.videos else []
        total_reward = 0.0
        while env.agents:
            actions = scripted_team_actions(env)
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
        print(json.dumps(result, ensure_ascii=False))
        if frames:
            imageio.mimsave(
                args.output / f"episode_{episode:03d}.mp4",
                frames,
                fps=env.metadata["render_fps"],
                macro_block_size=1,
            )

    summary = {
        "episodes": args.episodes,
        "success_rate": sum(item["success"] for item in results) / args.episodes,
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"success_rate={summary['success_rate']:.1%}; output={args.output}")


if __name__ == "__main__":
    main()
