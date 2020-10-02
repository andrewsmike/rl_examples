from random import choice, random

from gym.core import Env
from gym.utils import seeding
from gym.spaces import Discrete
import numpy as np
import matplotlib.pyplot as plt

def weighted_choice(choices, probs):
    value = random()
    for choice, prob in zip(choices, probs):
        value -= prob
        if value <= 0:
            return choice
    else:
        return choices[-1]

class GridEnv(Env):
    def __init__(
            self,
            start_state_dist=None,
            state_rewards=None,
            end_states=None,
            tinyworld_action_noise=True,
    ):
        self.start_state_dist = start_state_dist
        self.state_rewards = state_rewards
        self.end_states = end_states
        self.tinyworld_action_noise = tinyworld_action_noise

        self.states = sorted(
            state_rewards.keys(),
            key=lambda x_y: (x_y[1], x_y[0]),
        )
        self.state_indices = {
            state: state_index + 1
            for state_index, state in enumerate(self.states)
        }

        self.action_space = Discrete(4)
        self.observation_space = Discrete(len(self.states))

        self.seed()

        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.tinyworld_action_noise:
            action = weighted_choice(
                [action, (action + 1) % 4, (action + 3) % 4, None],
                [0.80, 0.05, 0.05, 0.10],
            )
        else:
            pass

        if action is not None:
            dx, dy = [
                (-1, 0), # Left
                (0, -1), # Up
                (1, 0), # Right
                (0, 1), # Down
            ][int(action)]

            x, y = self.state
            next_state = (x + dx), (y + dy)
            if next_state in self.state_rewards:
                self.state = next_state

        return (
            self.state_indices[self.state],
            self.state_rewards[self.state],
            (self.state in self.end_states),
            {},
        )

    def reset(self):
        states, probs = zip(*self.start_state_dist.items())
        self.state = states[weighted_choice(list(range(len(states))), probs)]

        return self.state_indices[self.state]

    def render(self, mode=None):
        display_grid_env(
            self.states,
            start_positions=set(self.start_state_dist.keys()),
            finish_positions=self.end_states,
            agent_position=self.state,
        )

    def render_agent_data(self, state_data, logger=None):
        state_data = {
            state: state_data.get(state_index + 1, "")
            for state_index, state in enumerate(self.states)
        }
        display_state_data(state_data, logger=logger)

def grid_world_state_rewards(grid_world):
    state_rewards = {}

    for y, world_row in enumerate(grid_world):
        for x, cell_reward in enumerate(world_row):
            if cell_reward is None:
                continue

            state_rewards[(x, y)] = cell_reward

    return state_rewards

tiny_grid_world = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, None, 0, 0],
    [0, 0, None, 0, 0],
    [0, 0, -10, 0, 10],
]

class TinyGridEnv(GridEnv):
    def __init__(self):
        super().__init__(
            start_state_dist={(0, 0): 1.0},
            state_rewards=grid_world_state_rewards(tiny_grid_world),
            end_states={(4, 4)},
            tinyworld_action_noise=True,
        )

        assert self.state_rewards[(4, 4)] == 10

cliff_grid_world = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1],
]

class CliffGridEnv(GridEnv):
    def __init__(self):
        super().__init__(
            start_state_dist={(0, 3): 1.0},
            state_rewards=grid_world_state_rewards(cliff_grid_world),
            end_states={(11, 3)},
            tinyworld_action_noise=False,
        )

        assert self.state_rewards[(11, 3)] == -1


class VelocityGridEnv(GridEnv):
    """
    5x5 velocities.
    """
    def __init__(
            self,
            positions=None,
            start_positions=None,
            end_positions=None,
    ):
        self.positions = positions
        self.start_positions = start_positions
        self.end_positions = end_positions

        self.position_index = {
            position: position_index
            for position_index, position in enumerate(positions)
        }
        self.velocity_index = {
            x_y: velocity_index
            for velocity_index, x_y in enumerate(
                    (x, -y)
                    for x in range(5)
                    for y in range(5)
            )
        }

        self.action_space = Discrete(9)
        self.observation_space = Discrete(
            len(positions) * 25,
        )

        self.seed()

    def state_index(self, position=None, velocity=None):
        if position is None:
            position = self.position
            velocity = self.velocity

        return (
            25 * self.position_index[position]
            + self.velocity_index[velocity]
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ddx, ddy = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ][int(action)]

        acc_choices = [(ddx, ddy), (0, 0)]
        ddx, ddy = acc_choices[weighted_choice(list(range(len(acc_choices))), [0.9, 0.1])]

        dx, dy = self.velocity

        next_dx, next_dy = min(max(dx + ddx, 0), 4), min(max(dy + ddy, -4), 0)
        if next_dx == next_dy == 0:
            next_dx, next_dy = dx, dy

        self.velocity = next_dx, next_dy

        x, y = self.position
        next_x, next_y = x + dx, y + dy
        if (next_x, next_y) not in self.positions:
            self.reset()
        else:
            self.position = x + dx, y + dy

        return (
            self.state_index(),
            -1,
            (self.position in self.end_positions),
            {},
        )

    def reset(self):
        self.position = self.start_positions[
            choice(list(range(len(self.start_positions))))
        ]
        self.velocity = (0, 0)

        return -1

    def render(self, mode=None):
        display_grid_env(
            self.positions,
            start_positions=self.start_positions,
            finish_positions=self.end_positions,
            agent_position=self.position,
        )

    def render_agent_data(self, state_data, logger=None):
        state_data = {
            position: max((
                state_data.get(
                    self.state_index(position, velocity),
                    "",
                )
                for velocity in (
                        (dx, -dy)
                        for dx in range(5)
                        for dy in range(5)
                )
                if self.state_index(position, velocity) in state_data
            ), default="")
            for position in self.positions
        }

        display_state_data(state_data, logger=logger)

racetrack_one_compact_world = """\
                FFFFF
                FFFFF
                FFFFF
   OOOOOOOOOOOOOFFFFF
  OOOOOOOOOOOOOOFFFFF
  OOOOOOOOOOOOOOFFFFF
 OOOOOOOOOOOOOOOFFFFF
OOOOOOOOOOOOOOOOFFFFF
OOOOOOOOOOOOOOOOFFFFF
OOOOOOOOOO      FFFFF
OOOOOOOOO       FFFFF
OOOOOOOOO       FFFFF
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO
 OOOOOOOO
 OOOOOOOO
 OOOOOOOO
 OOOOOOOO
 OOOOOOOO
 OOOOOOOO
 OOOOOOOO
 OOOOOOOO
  OOOOOOO
  OOOOOOO
  OOOOOOO
  OOOOOOO
  OOOOOOO
  OOOOOOO
  OOOOOOO
   OOOOOO
   OOOOOO
   SSSSSS
"""

def velocity_env_args(compact_world_str):
    positions = [
        (x, y)
        for y, world_row_str in enumerate(compact_world_str.split("\n"))
        for x, state_str in enumerate(world_row_str)
        if state_str != " "
    ]

    start_positions = [
        (x, y)
        for y, world_row_str in enumerate(compact_world_str.split("\n"))
        for x, state_str in enumerate(world_row_str)
        if state_str == "S"
    ]

    end_positions = [
        (x, y)
        for y, world_row_str in enumerate(compact_world_str.split("\n"))
        for x, state_str in enumerate(world_row_str)
        if state_str == "F"
    ]

    return {
        "positions": positions,
        "start_positions": start_positions,
        "end_positions": end_positions,
    }


class RacetrackOneGridEnv(VelocityGridEnv):
    def __init__(self):
        super().__init__(
            **velocity_env_args(racetrack_one_compact_world)
        )

racetrack_two_compact_world = """\
                               FFFFF
                               FFFFF
                               FFFFF
                               FFFFF
                OOOOOOOOOOOOOOOFFFFF
             OOOOOOOOOOOOOOOOOOFFFFF
            OOOOOOOOOOOOOOOOOOOFFFFF
           OOOOOOOOOOOOOOOOOOOOFFFFF
           OOOOOOOOOOOOOOOOOOOOFFFFF
           OOOOOOOOOOOOOOOOOOOOFFFFF
           OOOOOOOOOOOOOOOOOOOOFFFFF
            OOOOOOOOOOOOOOOOOOOFFFFF
             OOOOOOOOOOOOOOOOOOFFFFF
              OOOOOOOOOOOOOOOO FFFFF
              OOOOOOOOOOOOO    FFFFF
              OOOOOOOOOOOO     FFFFF
              OOOOOOOOOO       FFFFF
              OOOOOOOOO
             OOOOOOOOOO
            OOOOOOOOOOO
           OOOOOOOOOOOO
          OOOOOOOOOOOOO
         OOOOOOOOOOOOOO
        OOOOOOOOOOOOOOO
       OOOOOOOOOOOOOOOO
      OOOOOOOOOOOOOOOOO
     OOOOOOOOOOOOOOOOOO
    OOOOOOOOOOOOOOOOOOO
   OOOOOOOOOOOOOOOOOOOO
  OOOOOOOOOOOOOOOOOOOOO
 OOOOOOOOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOOOOO
SSSSSSSSSSSSSSSSSSSSSSS
"""

class RacetrackTwoGridEnv(VelocityGridEnv):
    def __init__(self):
        super().__init__(
            **velocity_env_args(racetrack_two_compact_world)
        )


def display_grid(grid, logger=None):
    if logger is not None:
        display = lambda message: logger.info(message)
    else:
        display = print

    height, width = len(grid), len(grid[0])

    box_width = max(len(str(value)) for row in grid for value in row)
    base = "+-" + ("-" * box_width) + "-"
    sep_line = base * width + "+"

    display(sep_line)
    for row in grid:
        display("| " + " | ".join([
            str(item).rjust(box_width)
            for item in row
        ]) + " |")
        display(sep_line)

def display_grid_compact(grid, logger=None):
    if logger is not None:
        display = lambda message: logger.info(message)
    else:
        display = print

    height, width = len(grid), len(grid[0])

    box_width = max(len(str(value)) for row in grid for value in row)

    for row in grid:
        display(" ".join([
            str(item).rjust(box_width)
            for item in row
        ]))


def display_state_data(state_data, logger=None):
    xs, ys = zip(*state_data.keys())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    display_grid_compact([
        [
            state_data.get((x, y), "###")
            for x in range(min_x, max_x + 1)
        ]
        for y in range(min_y, max_y + 1)
    ], logger=logger)

def position_pixel(
        position,
        positions,
        start_positions,
        finish_positions,
        agent_position,
):
    if position in start_positions:
        return [64, 64, 192]
    elif position in finish_positions:
        return [192, 64, 64]
    elif position == agent_position:
        return [64, 192, 64]
    elif position not in positions:
        return [0, 0, 0]
    else:
        return [192, 192, 192]

def display_grid_env(
        positions=None,
        start_positions=None,
        finish_positions=None,
        agent_position=None,
):
    display_color_grid({
        position: position_pixel(
            position,
            positions,
            start_positions,
            finish_positions,
            agent_position,
        )
        for position in positions
    })

def display_color_grid(
        position_color,
):
    xs, ys = zip(*position_color.keys())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    white_pixel = [0, 0, 0]

    plt.imshow(np.array([
        [
            position_color.get((x, y), white_pixel)
            for x in range(min_x, max_x + 1)
        ]
        for y in range(min_y, max_y + 1)
    ]))
    # plt.clf()
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    display_state_data(grid_world_state_rewards(tiny_grid_world))
    display_state_data({
        state: state_index + 1
        for state_index, state in enumerate(sorted(grid_world_state_rewards(tiny_grid_world).keys(), key=lambda x_y: (x_y[1], x_y[0])))
    })

