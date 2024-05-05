import math
from typing import List, Union
from matrix import Matrix


MAX_ITERATIONS = 100


def vector_difference(a: List[Union[int, float]],
                      b: List[Union[int, float]]) -> List[Union[int, float]]:
    return [a[i] - b[i] for i in range(len(a))]


def jacobi_method(A: Matrix, b):

    x = [1 for _ in range(A.dims[0])]

    res = vector_difference(A * x, b)

    iterations = 0
    error = [math.sqrt(sum(res[i] ** 2 for i in range(A.dims[0])))]

    while error[-1] > 1e-6 and iterations < MAX_ITERATIONS:
        iterations += 1
        new_x = [0 for _ in range(len(x))]
        for i in range(A.dims[0]):
            new_x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(A.dims[0]) if j != i)) / A[i, i]
        x = new_x
        res = vector_difference(A * x, b)
        error.append(math.sqrt(sum(res[i] ** 2 for i in range(A.dims[0]))))

    return x, iterations, error


def gauss_seidel_method(A: Matrix, b):

    x = [1 for _ in range(A.dims[0])]

    res = vector_difference(A * x, b)

    iterations = 0
    error = [math.sqrt(sum(res[i] ** 2 for i in range(A.dims[0])))]
    while error[-1] > 1e-6 and iterations < MAX_ITERATIONS:
        iterations += 1
        for i in range(A.dims[0]):
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(i)) - sum(A[i, j] * x[j] for j in range(i+1, A.dims[0]))) / A[i, i]
        
        res = vector_difference(A * x, b)
        error.append(math.sqrt(sum(res[i] ** 2 for i in range(A.dims[0]))))

    return x, iterations, error
