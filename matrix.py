from typing import List, Union


class Matrix:

    def __init__(self, dims):
        x, y = dims
        self.dims = dims
        self.data = [[0 for j in range(x)] for i in range(y)]

    def __getitem__(self, key):
        i, j = key
        return self.data[i][j]

    def __setitem__(self, key, value):
        i, j = key
        self.data[i][j] = value

    def __add__(self, other: 'Matrix') -> 'Matrix':

        if self.dims != other.dims:
            raise ValueError("Matrix dimensions do not match")

        x, y = self.dims
        result = Matrix(self.dims)

        for i in range(y):
            for j in range(x):
                result[i, j] = self[i, j] + other[i, j]

        return result

    def __mul__(self, other) -> 'Matrix':

        if isinstance(other, Matrix):
            return Matrix.matmul(self, other)

        if type(other) is list:
            return Matrix.matvecmul(self, other)

        raise TypeError("Matrix multiplication is not defined for this type")

    def __repr__(self) -> str:
        return '\n'.join([' '.join(map(str, row)) for row in self.data])

    @staticmethod
    def from_function(dims, func):
        x, y = dims
        matrix = Matrix(dims)

        for i in range(y):
            for j in range(x):
                matrix[i, j] = func(i, j)

        return matrix

    @staticmethod
    def matmul(a, b):
        x, y = a.dims
        z, w = b.dims

        if y != z:
            raise ValueError("Matrix dimensions do not match")

        result = Matrix((w, x))

        for i in range(w):
            for j in range(x):
                for k in range(y):
                    result[i, j] += a[k, j] * b[i, k]

        return result

    @staticmethod
    def matvecmul(a: 'Matrix', b: List[Union[int, float]]) -> List[float]:
        x, y = a.dims

        if y != len(b):
            raise ValueError("Matrix dimensions do not match")

        result = [0 for _ in range(x)]

        for i in range(x):
            for j in range(y):
                result[i] += a[i, j] * b[j]

        return result
