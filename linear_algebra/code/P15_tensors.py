import numpy as np
rng = np.random.default_rng(0)      # random number generator for test matrices


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


def test():
    T = np.arange(24).reshape(2, 3, 4)
    flat = T.reshape(-1)
    print("index formula ok:", all(T[i, j, k] == flat[12 * i + 4 * j + k]
                                   for i in range(2) for j in range(3) for k in range(4)))
    U1 = T.reshape(2, 12)
    U2 = T.transpose(1, 0, 2).reshape(3, 8)
    print("ranks of unfoldings:", np.linalg.matrix_rank(U1), np.linalg.matrix_rank(U2))

    A, B = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    x, y = rng.normal(size=4), rng.normal(size=4)
    checks = [
        np.allclose(np.einsum("ij,jk->ik", A, B), A @ B),
        np.isclose(np.einsum("ii->", A), np.trace(A)),
        np.isclose(np.einsum("i,i->", x, y), x @ y),
        np.allclose(np.einsum("i,j->ij", x, y), np.outer(x, y)),
        np.allclose(np.einsum("ij->ji", A), A.T),
        np.isclose(np.einsum("i,ij,j->", x, A, y), x @ A @ y),
    ]
    print("Table 1 checks:", checks)

    As, Bs = rng.normal(size=(100, 4, 4)), rng.normal(size=(100, 4, 4))
    tr = np.einsum("bii->b", As)
    prod = np.einsum("bij,bjk->bik", As, Bs)
    print("batch traces ok:", np.allclose(tr, [np.trace(M) for M in As]))
    print("batch products ok:", np.allclose(prod, np.array([a @ b for a, b in zip(As, Bs)])))

    A, B, X = rng.normal(size=(3, 4)), rng.normal(size=(2, 5)), rng.normal(size=(4, 5))
    print("kron identity ok:", np.allclose((A @ X @ B.T).reshape(-1), np.kron(A, B) @ X.reshape(-1)))

    u, v, w = rng.normal(size=3), rng.normal(size=4), rng.normal(size=2)
    R1 = rank_one(u, v, w)
    print("slices are w_k u v^T:", all(np.allclose(R1[:, :, k], w[k] * np.outer(u, v)) for k in range(2)))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test()
