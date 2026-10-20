import numpy as np
from pettingzoo.test import parallel_api_test

from airport_cargo_mawm.envs.airport_cargo import AirportCargoEnv
from airport_cargo_mawm.policies import scripted_team_actions


def test_parallel_api() -> None:
    parallel_api_test(AirportCargoEnv(max_steps=12), num_cycles=30)


def test_state_and_render_shapes() -> None:
    env = AirportCargoEnv(image_size=96)
    observations, _ = env.reset(seed=7)
    assert observations["tug_0"].shape == (11,)
    assert env.global_state().shape == (11,)
    assert env.render().shape == (96, 96, 3)
    assert env.render().dtype == np.uint8


def test_scripted_team_solves_task() -> None:
    for seed in range(5):
        env = AirportCargoEnv(max_steps=80)
        env.reset(seed=seed)
        while env.agents:
            env.step(scripted_team_actions(env))
        assert env.success, f"scripted team failed for seed {seed}"
