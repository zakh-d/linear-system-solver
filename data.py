from typing import Tuple
from math import sin

from tensors import Matrix, Vector


def generate_A(dimentions: Tuple[int, int], a1, a2, a3) -> Matrix:
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


def generate_b(N: int) -> Vector:
    return Vector([sin(7*n) for n in range(N)], is_column=True)
