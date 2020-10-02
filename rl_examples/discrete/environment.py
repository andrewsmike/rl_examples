from copy import deepcopy
from random import choices

def sample_dist(dist):
    outcomes, probabilities = zip(*dist.items())
    assert sum(probabilities) == 1
    return choices(outcomes, probabilities, k=1)[0]

class Environment():

    def start_state(self):
        """
        s_0
        """
        raise NotImplementedError

    def states(self):
        """
        S
        """
        raise NotImplementedError

    def state_actions(self, state):
        """
        ACTIONS(s)
        """
        raise NotImplementedError
        
    def state_action_result_dist(self, state, action):
        """
        P(s' | s, a)
        """
        raise NotImplementedError

    def sample_state_action_result(self, state, action):
        """
        s' ~ P(s' | s, a)
        """
        dist = self.state_action_result_dist(state, action)
        return sample_dist(dist)

    def state_is_terminal(self, state):
        """
        TERMINAL(s)
        """
        raise NotImplementedError

    def state_reward(self, state):
        """
        R(s)
        """
        raise NotImplementedError

    def discount(self):
        """
        \gamma
        """
        raise NotImplementedError

    def max_reward(self):
        """
        R_{max}
        """
        raise NotImplementedError

    def min_reward(self):
        """
        R_{min}
        """
        raise NotImplementedError


grid_world_directions = ["LEFT", "UP", "RIGHT", "DOWN"]
grid_world_direction_deltas = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
grid_world_actions = set(grid_world_directions)

# 10% of accidentally going left, 10% of going right of intended direction.
grid_world_move_noise = 0.1

def add_pos(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return x1 + x2, y1 + y2

def left_of(direction):
    return grid_world_directions[
        (grid_world_directions.index(direction) + 3) % 4
    ]

def right_of(direction):
    return grid_world_directions[
        (grid_world_directions.index(direction) + 1) % 4
    ]

def random_grid_world(width, height):
    raise NotImplemetedError

def example_grid_world(empty_reward=-0.04):
    r = empty_reward
    return (
        (0, 0),
        [[("EMPTY", r), ("EMPTY", r), ("EMPTY", r), ( "EXIT",  1)],
         [("EMPTY", r), ( "WALL", 0), ("EMPTY", r), ( "EXIT", -1)],
         [("EMPTY", r), ("EMPTY", r), ("EMPTY", r), ("EMPTY",  r)]],
    )

def movement_outcome_dist(start_pos, direction):
    chance = grid_world_move_noise

    actual_dir_dist = {
        left_of(direction): (chance),
        right_of(direction): (chance),
        direction: (1 - 2 * chance),
    }

    return {
        add_pos(start_pos, grid_world_direction_deltas[dir]): p
        for dir, p in actual_dir_dist.items()
    }

class GridWorldEnvironment(Environment):
    def __init__(self, gamma=0.9):
        """
        Grid world is indexed as [y][x].
        """
        self.width, self.height = 4, 3
        self.start_space, self.grid_world = example_grid_world()
        self.gamma = gamma

    def start_state(self):
        """
        s_0
        """
        return self.start_space

    def states(self):
        """
        S
        """
        return {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
        }

    def state_actions(self, state):
        """
        ACTIONS(s)
        """
        x, y = state
        if self.grid_world[y][x][0] in ("EXIT", "WALL"):
            return set()
        else:
            return grid_world_actions

    def state_action_result_dist(self, state, action):
        """
        P(s' | s, a).
        """
        if action is None or not self.state_actions(state):
            return {state: 1.0}

        next_state_dist = movement_outcome_dist(state, action)

        # Cancel movement when we're gonna hit a wall.
        outcome_probability = {}
        for next_state, next_state_p in next_state_dist.items():
            next_x, next_y = next_state
            next_state_blocked = (
                (not (0 <= next_x < self.width)) or
                (not (0 <= next_y < self.height)) or
                (self.grid_world[next_y][next_x][0] == "WALL")
            )
            if next_state_blocked:
                if state not in outcome_probability:
                    outcome_probability[state] = 0

                outcome_probability[state] += next_state_p
            else:
                if next_state not in outcome_probability:
                    outcome_probability[next_state] = 0

                outcome_probability[next_state] += next_state_p

        assert sum(outcome_probability.values()) == 1
        return outcome_probability
        
    def state_is_terminal(self, state):
        """
        TERMINAL(s)
        """
        x, y = state
        return self.grid_world[y][x][0] == "EXIT"

    def state_reward(self, state):
        """
        R(s)
        """
        x, y = state
        return self.grid_world[y][x][1]

    def discount(self):
        """
        \gamma
        """
        return self.gamma

    def max_reward(self):
        """
        R_{max}
        """
        return max(reward for row in self.grid_world for tile, reward in row)

    def min_reward(self):
        """
        R_{min}
        """
        return min(reward for row in self.grid_world for tile, reward in row)


    def display_trace(self, trace, display="rewards"):
        grid = [
            [
                column_str(column, display, trace)
                for column in row
            ]
            for row in self.grid_world
        ]
        
        for 
        # [y][x]
        self.width, self.height
        display_grid([
            [
                grid[row_index][column_index]
                for column_index in range(self.width)
            ]
            for row_index in range(self.height)
        ])

def column_str(column, display, trace):
    if display == "rewards":
        if 


def display_grid(grid):
    height, width = len(grid), len(grid[0])

    box_width = max(len(str(value)) for row in grid for value in row)
    base = "+-" + ("-" * box_width) + "-"
    sep_line = base * width + "+"

    print(row_sep_line)
    for row in grid:
        print("| " + " | ".join(map(str, row)) + " |")
        print(row_sep_line)
