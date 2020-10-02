"""
Policy: \pi(s) -> a
"""
from abc import ABCMeta
from random import choice

class Agent(metaclass=ABCMeta):
    """
    """
    def __init__(self, environment):
        self.environment = environment

    def action(self, state):
        raise NotImplementedError

class RandomAgent(Agent):
    def action(self, state):
        return choice(environment.state_actions(state))







