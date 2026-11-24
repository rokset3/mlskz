import numpy as np


rng = np.random.default_rng(0)

def rref(A, tol=1e-10):
    """Return (R, pivots): the RREF of A and the list of pivot columns."""
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        p = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[p, c]) < tol:
            R[r:, c] = 0.0
            continue
        R[[r, p]] = R[[p, r]]
        R[r] = R[r] / R[r, c]
        for i in range(m):
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def rank(A):
    return len(rref(A)[1])


def null_space_basis(A):
    R, piv = rref(A)
    n = R.shape[1]
    free = [j for j in range(n) if j not in piv]
    basis = []
    for f in free:
        v = np.zeros(n)
        v[f] = 1.0
        for i, p in enumerate(piv):
            v[p] = -R[i, f]
        basis.append(v)
    return basis


def col_space_basis(A):
    A = np.array(A, dtype=float)
    _, piv = rref(A)
    return [A[:, j] for j in piv]


def test():
    print("=== P3 ===")
    R, piv = rref([[1, 2, -1], [3, -1, 4]])
    print(R, piv)
    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(rref(F)[0])
    ok = True
    for _ in range(20):
        A = rng.normal(size=(5, 7))
        N = null_space_basis(A)
        ok &= all(np.allclose(A @ v, 0) for v in N)
        ok &= rank(A) + len(N) == 7
        ok &= rank(A) == np.linalg.matrix_rank(A)
    print("random 5x7 tests passed:", ok)
    A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    print("rank of 6x8 product:", rank(A), " dim null:", len(null_space_basis(A)))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test()
