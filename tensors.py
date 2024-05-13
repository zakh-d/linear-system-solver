from math import sqrt
from typing import List, Union


class Vector:

    def __init__(self, data: Union[List, int], is_column=True):

        if type(data) is list:
            self.data = data
        elif type(data) is int:
            self.data = [0 for _ in range(data)]
        else:
            raise TypeError("Vector must be initialized with a list or an integer")
        self.is_column = is_column

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other: 'Vector') -> 'Vector':
        if len(self.data) != len(other.data) or self.is_column != other.is_column:
            raise ValueError("Vector dimensions do not match")
        return Vector([self[i] + other[i] for i in range(len(self.data))], self.is_column)

    def __sub__(self, other: 'Vector') -> 'Vector':
        if len(self.data) != len(other.data) or self.is_column != other.is_column:
            raise ValueError("Vector dimensions do not match")
        return Vector([self[i] - other[i] for i in range(len(self.data))], self.is_column)

    def __len__(self):
        return len(self.data)

    def euclidian_norm(self):
        return sqrt(sum(self[i] ** 2 for i in range(len(self.data))))


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

        if isinstance(other, Vector):
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

        result = Matrix((x, w))

        for i in range(x):
            for j in range(w):
                for k in range(y):
                    result[i, j] += a[i, k] * b[k, j]

        return result

    @staticmethod
    def matvecmul(a: 'Matrix', b: Vector) -> Union['Matrix', Vector]:
        rows, cols = a.dims

        # row vector
        if not b.is_column:
            matrix_from_b = Matrix.from_function((1, len(b)), lambda i, j: b[j])
            return Matrix.matmul(a, matrix_from_b)

        # column vector
        if cols != len(b):
            raise ValueError("Matrix dimensions do not match")

        result = Vector(rows, is_column=True)

        for i in range(rows):
            for j in range(cols):
                result[i] += a[i, j] * b[j]

        return result

    @staticmethod
    def eye(size):
        return Matrix.from_function((size, size), lambda i, j: 1 if i == j else 0)
