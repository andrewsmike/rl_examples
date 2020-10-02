"""
Discritize continuous spaces.
This allows us to treat continuous problems as if they were discrete.
"""
from abc import abstractmethod, ABCMeta
from operator import mul
from typing import Any, List, Tuple

from gym.spaces import flatten, flatten_space, Box, Discrete, Space, Tuple as TupleSpace
import numpy as np

class Discritizer(object, metaclass=ABCMeta):

    def __init__(self, input_space : Space):
        self.input_space = input_space

    @abstractmethod
    def output_space(self) -> Space:
        pass

    @abstractmethod
    def discritized(self, point : Any) -> Any:
        pass

class GridDiscritizer(Discritizer):

    def __init__(
            self,
            input_space : Space,
            increment_count : int = 25,
    ):
        super().__init__(input_space)
        assert isinstance(input_space, Box)

        self.increment_count = increment_count

        flattened_space = flatten_space(input_space)
        self.lows = flattened_space.low
        self.highs = flattened_space.high

        self.size = flattened_space.shape[0]
        assert self.size < 100000, (
            "Space too large to discritize. Size: {self.size}"
        )


    def output_space(self) -> Space:
        return TupleSpace(tuple(
            Discrete(self.increment_count)
            for dim_index in range(self.size)
        ))

    def discritized(
            self,
            point : np.ndarray
    ) -> Tuple[int]:
        flattened_point = point.reshape(-1)
        return tuple(np.floor(np.clip(
            self.increment_count * (flattened_point - self.lows)
            / (self.highs - self.lows),
            0,
            self.increment_count - 1,
        )).astype(int))

class CartPoleWideDiscritizer(GridDiscritizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lows = np.array([-4.8, -10, -0.419, -10])
        self.highs = np.array([4.8, 10, 0.419, 10.0])

class CartPoleDiscritizer(GridDiscritizer):
    def __init__(self, *args, increment_count=16, **kwargs):
        super().__init__(*args, increment_count=increment_count, **kwargs)
        self.lows = np.array([-1, -3, -0.2, -3])
        self.highs = np.array([1, 3, 0.2, 3])



discritizer_name_func = {
    "grid": GridDiscritizer,
    "cartpole": CartPoleDiscritizer,
    "cartpole-wide": CartPoleWideDiscritizer,
}
