"""
Baseline agents, including constant, random, sticky-random, and sticky-manual.
"""
from abc import abstractmethod, ABCMeta
import logging
from pprint import pprint
from random import choice
from typing import Optional

from gym import Env
from gym.spaces import Space

class Agent(metaclass=ABCMeta):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            trace_length: Optional[int] = None,
            gamma : float = 0.98,
            env : Optional[Env] = None,
    ):
        super().__init__()

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space

        self.gamma = gamma

        self.trace_length = trace_length

        self.reset_params()

    def reset_params(self):
        self.trace = []

    def observe(self, observation, action, reward):
        """
        Observe S_t, A_t, R_t.
        """
        if self.trace_length is not None:
            self.trace.append((observation, action, reward))
            if self.trace_length != -1:
                self.trace = self.trace[-self.trace_length:]

    @abstractmethod
    def sample_action(self, observation):
        pass

    def reset(self):
        self.trace = []

    def render_agent_data(self, state_data):
        env = self.env
        if env is not None and hasattr(env, "env"):
            env = self.env.env

        if hasattr(env, "render_agent_data"):
            env.render_agent_data(state_data)
        else:
            pprint(state_data)
    

class RandomAgent(Agent):

    def sample_action(self, observation):
        """
        Implicitly uses the agent's history as an argument.
        """
        return self.action_space.sample()

class ConstantAgent(Agent):
    def __init__(
            self,
            trace_length: Optional[int] = None,
            *args,
            **kwargs,
    ):
        super().__init__(
            trace_length=trace_length,
            *args,
            **kwargs,
        )

    def reset_params(self):
        super().reset_params()

        self.constant_action = self.action_space.sample()

    def sample_action(self, observation):
        return self.constant_action

    def reset(self):
        super().reset()

        self.reset_params()


class StickyRandomAgent(Agent):
    def __init__(
            self,
            stick_steps: int = 40,
            trace_length: Optional[int] = None,
            *args,
            **kwargs,
    ):
        self.stick_steps = stick_steps

        super().__init__(
            trace_length=trace_length,
            *args,
            **kwargs,
        )

    def reset_params(self):
        super().reset_params()

        self.remaining_steps = self.stick_steps
        self.current_action = self.action_space.sample()

    def observe(self, observation, action, reward):
        self.remaining_steps -= 1
        if self.remaining_steps == 0:
            self.reset_params()

    def reset(self):
        super().reset()

        self.reset_params()

    def sample_action(self, observation):
        return self.current_action

class StickyManualAgent(Agent):
    def __init__(
            self,
            stick_steps: int = 20,
            trace_length: Optional[int] = None,
            *args,
            **kwargs,
    ):
        self.stick_steps = stick_steps

        super().__init__(
            trace_length=trace_length,
            *args,
            **kwargs,
        )

    def reset_params(self):
        super().reset_params()

        self.remaining_steps = self.stick_steps
        self.current_action = int(input())

    def reset(self):
        super().reset()

        self.reset_params()

    def observe(self, observation, action, reward):
        self.remaining_steps -= 1
        if self.remaining_steps == 0:
            self.reset_params()

    def sample_action(self, observation):
        return self.current_action

