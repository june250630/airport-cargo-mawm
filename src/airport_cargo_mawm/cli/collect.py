from __future__ import annotations

import argparse
import json
from pathlib import Path

from airport_cargo_mawm.data import collect_scripted_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect scripted airport-cargo data.")
    parser.add_argument("--output", type=Path, default=Path("data/airport_cargo.h5"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--noise", type=float, default=0.03)
    args = parser.parse_args()
    summary = collect_scripted_dataset(
        args.output,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        noise=args.noise,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
