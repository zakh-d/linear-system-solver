from typing import Tuple, List
from math import sin

from matrix import Matrix


def generate_A(dimentions: Tuple[int, int], a1, a2, a3) -> List[List[int]]:
    x, y = dimentions

    def get_element(i: int, j: int):
        if i == j:
            return a1
        if abs(i - j) == 1:
            return a2
        if abs(i-j) == 2:
            return a3
        return 0

    return Matrix.from_function(dimentions, get_element)


def generate_b(N: int) -> List[int]:
    return [sin(7*n) for n in range(N)]
