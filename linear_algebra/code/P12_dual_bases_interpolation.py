import numpy as np
rng = np.random.default_rng(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dual_basis(P):
    return np.linalg.inv(P)


def vandermonde(t):
    t = np.asarray(t, dtype=float)
    return t[:, None] ** np.arange(len(t))[None, :]


def lagrange_coeffs(t):
    return np.linalg.inv(vandermonde(t))


def interpolate(f, t, xs):
    C = lagrange_coeffs(t)
    coeffs = C @ f(np.asarray(t))
    return np.polyval(coeffs[::-1], xs)


def test():
    P = rng.normal(size=(5, 5))
    Rw = dual_basis(P)
    print("r_i b_j = I:", np.allclose(Rw @ P, np.eye(5)))
    
    x = rng.normal(size=5)
    print("x = sum (r_i x) b_i:", np.allclose(sum((Rw[i] @ x) * P[:, i] for i in range(5)), x))
    print("Lagrange coefficients for nodes -1,0,1 (cols = L_-1, L_0, L_1):")
    print(lagrange_coeffs([-1, 0, 1]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    t = np.linspace(-1, 1, 9)
    xs = np.linspace(-1, 1, 400)
    f = lambda s: np.sin(3 * s)
    
    axes[0].plot(xs, f(xs), label="sin(3t)")
    axes[0].plot(xs, interpolate(f, t, xs), "--", label="interpolant, 9 nodes")
    axes[0].plot(t, f(t), "ko", ms=3)
    axes[0].legend()
    
    t = np.linspace(-5, 5, 21)
    xs = np.linspace(-5, 5, 800)
    g = lambda s: 1 / (1 + s ** 2)
    
    axes[1].plot(xs, g(xs), label="1/(1+t^2)")
    axes[1].plot(xs, interpolate(g, t, xs), "--", label="interpolant, 21 nodes")
    axes[1].plot(t, g(t), "ko", ms=3)
    axes[1].set_ylim(-1, 2)
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig("p12_interpolation.png", dpi=130)
    err = np.max(np.abs(interpolate(g, t, xs) - g(xs)))
    print("max error Runge example:", err)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test()
