import numpy as np


def forward_substitution(L, b):
    """Solve Lx=b for a lower-triangular matrix L."""
    n, _ = L.shape
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x


def backward_substitution(U, b):
    """Solve Ux=b for an upper-triangular matrix U."""
    n, _ = U.shape
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        if abs(U[i, i]) < 1e-10:
            x[i] = 0.0
        else:
            x[i] /= U[i, i]
    return x


def solve_cholesky(A, b, eps=1e-14):
    """Solve Ax=b using an explicit Cholesky factorization A=LL^T."""
    n, _ = A.shape
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            subtotal = 0.0
            for k in range(j):
                subtotal += L[i, k] * L[j, k]

            if i == j:
                value = A[i, i] - subtotal
                if value <= eps:
                    raise ValueError(
                        "Matrix is not positive definite (or is numerically ill-conditioned)."
                    )
                L[i, j] = np.sqrt(value)
            else:
                L[i, j] = (A[i, j] - subtotal) / L[j, j]

    y = forward_substitution(L, b)
    return backward_substitution(L.T, y)


def determinant(A):
    """Compute a determinant recursively by Laplace expansion."""
    n = len(A[:, 0])
    if n == 1:
        return A[0, 0]

    result = 0.0
    sign = 1.0
    for j in range(n):
        B = A[:, 1:n]
        B = B[[i for i in range(n) if i != j], :]
        result += A[j, 0] * sign * determinant(B)
        sign *= -1.0
    return result


def solve_cramer(A, b):
    """Solve Ax=b with Cramer's rule (intended only for small systems)."""
    det_A = determinant(A)
    if abs(det_A) < 1e-14:
        raise ValueError("Singular matrix: Cramer's rule cannot be applied.")

    x = np.zeros_like(b, dtype=float)
    for i in range(len(b)):
        B = A.astype(float).copy()
        B[:, i] = b
        x[i] = determinant(B) / det_A
    return x
