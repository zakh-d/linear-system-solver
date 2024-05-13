import copy
from typing import List, Tuple, Union
from tensors import Matrix, Vector


MAX_ITERATIONS = 100


def vector_difference(a: List[Union[int, float]],
                      b: List[Union[int, float]]) -> List[Union[int, float]]:
    return [a[i] - b[i] for i in range(len(a))]


def jacobi_method(A: Matrix, b: Vector):

    x = Vector([1 for _ in range(A.dims[0])])

    res = A * x - b

    iterations = 1
    error = [res.euclidian_norm()]

    while error[-1] > 1e-9 and iterations < MAX_ITERATIONS:
        iterations += 1
        new_x = Vector(len(x))
        for i in range(A.dims[0]):
            new_x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(A.dims[0]) if j != i)) / A[i, i]
        x = new_x
        res = A * x - b
        error.append(res.euclidian_norm())

    return x, iterations, error


def gauss_seidel_method(A: Matrix, b):

    x = Vector([1 for _ in range(A.dims[0])])

    res = A * x - b

    iterations = 1
    error = [res.euclidian_norm()]
    while error[-1] > 1e-9 and iterations < MAX_ITERATIONS:
        iterations += 1
        for i in range(A.dims[0]):
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(i)) - sum(A[i, j] * x[j] for j in range(i+1, A.dims[0]))
                    ) / A[i, i]

        res = A * x - b
        error.append(res.euclidian_norm())

    return x, iterations, error


def lu_factorization(A: Matrix) -> Tuple[Matrix, Matrix]:

    m = A.dims[0]

    U = copy.deepcopy(A)
    L = Matrix.eye(m)

    for i in range(2, m + 1):
        for j in range(1, i):
            L[i-1, j-1] = U[i-1, j-1] / U[j-1, j-1]
            for k in range(0, m):
                U[i-1, k] -= L[i-1, j-1] * U[j-1, k]
    return L, U


def direct_method(A: Matrix, b: Vector) -> Tuple[Vector, float]:
    L, U = lu_factorization(A)

    y = Vector(A.dims[0])

    # forward substitution for Ly = b
    for i in range(A.dims[0]):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    x = Vector(A.dims[0])

    # backward substitution for Ux = y
    for i in range(A.dims[0] - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, A.dims[0]))) / U[i, i]

    return x, (A*x - b).euclidian_norm()
