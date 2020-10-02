from rl_examples.envs.grid import TinyGridEnv, RacetrackOneGridEnv, RacetrackTwoGridEnv
from gym.envs import register

name_grid_env = {
    "tiny_grid": TinyGridEnv,
    "racetrack_one_grid": RacetrackOneGridEnv,
    "racetrack_two_grid": RacetrackTwoGridEnv,
}

register(
    id="TinyGrid-v0",
    entry_point="rl_examples.envs.grid:TinyGridEnv",
    max_episode_steps=200,
)

register(
    id="CliffGrid-v0",
    entry_point="rl_examples.envs.grid:CliffGridEnv",
    max_episode_steps=200,
)

register(
    id="RacetrackOneGrid-v0",
    entry_point="rl_examples.envs.grid:RacetrackOneGridEnv",
    max_episode_steps=200,
)

register(
    id="RacetrackTwoGrid-v0",
    entry_point="rl_examples.envs.grid:RacetrackTwoGridEnv",
    max_episode_steps=200,
)
