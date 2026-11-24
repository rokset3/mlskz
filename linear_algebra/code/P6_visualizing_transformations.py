import numpy as np
rng = np.random.default_rng(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


F_LETTER = np.array([(0.2, 0.1), (0.4, 0.1), (0.4, 0.45), (0.65, 0.45), (0.65, 0.6),
                     (0.4, 0.6), (0.4, 0.75), (0.8, 0.75), (0.8, 0.9), (0.2, 0.9)])
SQUARE = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)


def closed(P):
    return np.vstack([P, P[:1]])


def plot_transformation(A, ax=None, title=""):
    
    A = np.array(A, dtype=float)
    if ax is None:
        _, ax = plt.subplots()

    for t in np.linspace(0, 1, 5):
        for line in (np.array([(t, 0), (t, 1)]), np.array([(0, t), (1, t)])):
            ax.plot(*line.T, color="lightgray", lw=0.7)
            ax.plot(*(line @ A.T).T, color="lightblue", lw=0.7)
    
    ax.plot(*closed(SQUARE).T, "k--", lw=1)
    ax.fill(*(SQUARE @ A.T).T, color="tab:blue", alpha=0.15)
    ax.plot(*closed(SQUARE @ A.T).T, color="tab:blue", lw=1.5)
    ax.fill(*F_LETTER.T, color="gray", alpha=0.6)
    ax.fill(*(F_LETTER @ A.T).T, color="tab:orange", alpha=0.8)
    ax.arrow(0, 0, *A[:, 0], color="red", width=0.01, length_includes_head=True)
    ax.arrow(0, 0, *A[:, 1], color="green", width=0.01, length_includes_head=True)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    return ax


def signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test():
    mats = {"stretch": [[2, 0], [0, 1]], "rotation 90": [[0, -1], [1, 0]],
            "reflection": [[1, 0], [0, -1]], "shear": [[1, 1], [0, 1]],
            "singular": [[1, 2], [2, 4]], "Figure 4": [[3, 1], [1, 2]]}
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (name, A) in zip(axes.ravel(), mats.items()):
        plot_transformation(A, ax, name)
        A = np.array(A, dtype=float)
        print(f"{name:12s} shoelace = {signed_area(SQUARE @ A.T):6.2f}   det = {np.linalg.det(A):6.2f}")
    fig.tight_layout()
    fig.savefig("p6_transforms.png", dpi=130)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test()
