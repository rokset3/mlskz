import numpy as np
rng = np.random.default_rng(0)

import time


def lu(A):
    S = np.array(A, dtype=float)
    n = S.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        U[k, k:] = S[k, k:]
        L[k + 1:, k] = S[k + 1:, k] / S[k, k]
        S[k + 1:, k + 1:] -= np.outer(L[k + 1:, k], S[k, k + 1:])
    return L, U


def solve_lu(L, U, b):
    n = len(b)
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    
    for i in reversed(range(n)):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def sherman_morrison(Ainv, u, v):
    Au = Ainv @ u
    vA = v @ Ainv
    return Ainv - np.outer(Au, vA) / (1 + v @ Au)


def test_P16():
    L, U = lu([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    print("L =\n", L, "\nU =\n", U)
    
    ok = True
    for _ in range(10):
        A = rng.normal(size=(6, 6))
        L, U = lu(A)
        b = rng.normal(size=6)
        ok &= np.allclose(L @ U, A) and np.allclose(solve_lu(L, U, b), np.linalg.solve(A, b))
    print("random 6x6 tests passed:", ok)

    A = np.array([[1e-17, 1], [1, 1]])
    b = np.array([1.0, 2.0])
    L, U = lu(A)
    print("no pivoting:", solve_lu(L, U, b))
    
    L, U = lu(A[[1, 0]])
    print("rows swapped:", solve_lu(L, U, b[[1, 0]]))

    X = rng.normal(size=(200, 50))
    new_rows = rng.normal(size=(100, 50))
    t0 = time.perf_counter()
    M = np.linalg.inv(X.T @ X)
    
    for r in new_rows:
        M = sherman_morrison(M, r, r)
    t_sm = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xf = X.copy()
    
    for r in new_rows:
        Xf = np.vstack([Xf, r])
        M2 = np.linalg.inv(Xf.T @ Xf)
    t_scratch = time.perf_counter() - t0
    print("Sherman-Morrison matches:", np.allclose(M, M2))
    print(f"time SM: {t_sm*1000:.2f} ms, from scratch: {t_scratch*1000:.2f} ms")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test_P16()
