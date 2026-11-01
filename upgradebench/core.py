"""A latent-score model of a soft-voting blend, for asking how much of a member's solo gain reaches the blend.

Every member m scores every class c of every row as

    z_m[c] = a_m * (c == y) + sqrt(rho_m) * S[c] + sqrt(1 - rho_m) * E_m[c]

with S shared by all members (the mistakes they have in common) and E_m private. All noises are unit Gaussians, so a
member's solo accuracy depends only on its signal a_m and the number of classes, and rho_m only sets how its errors
line up with the others': corr(z_m, z_n) = sqrt(rho_m * rho_n) off the signal.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

CLASSES = 18


def solo_accuracy(a: float, classes: int = CLASSES, grid: int = 4001) -> float:
    """P(a + e_true > max of the other classes' noise), by numerical integration over e_true."""
    x = np.linspace(a - 8, a + 8, grid)
    pdf = np.exp(-0.5 * (x - a) ** 2) / np.sqrt(2 * np.pi)
    return float(np.trapezoid(pdf * ndtr(x) ** (classes - 1), x))


def signal_for(accuracy: float, classes: int = CLASSES) -> float:
    """The signal a whose solo accuracy is the given one (bisection; accuracy rises with a)."""
    lo, hi = 0.0, 12.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if solo_accuracy(mid, classes) < accuracy:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def one_hot(n: int, classes: int = CLASSES) -> np.ndarray:
    """Rows with balanced true class 0..classes-1; the signal is added on this matrix."""
    out = np.zeros((n, classes), dtype=np.float32)
    out[np.arange(n), np.arange(n) % classes] = 1
    return out


def member(signal: float, rho: float, onehot: np.ndarray, shared: np.ndarray, private: np.ndarray) -> np.ndarray:
    return signal * onehot + np.sqrt(rho) * shared + np.sqrt(1 - rho) * private


def correct(scores: np.ndarray) -> np.ndarray:
    """True where the top score is the true class (classes are balanced, so accuracy = balanced accuracy)."""
    return scores.argmax(1) == np.arange(len(scores)) % scores.shape[1]


def blend(rest: np.ndarray, target: np.ndarray, share: float) -> np.ndarray:
    """Soft vote: the target member holds `share` of the total weight, the rest of the blend the remainder."""
    return (1 - share) * rest + share * target


def noise_of_mean(k: int, rho: float) -> float:
    """Sd of the noise of a mean of k equal members: blends must be standardised before weights mean anything."""
    return float(np.sqrt(rho + (1 - rho) / k))


def detection(before: np.ndarray, after: np.ndarray, n_public: int = 1200) -> dict[str, float]:
    """Paired comparison of two right/wrong vectors on a public split of n_public rows."""
    delta = float(after.mean() - before.mean())
    flip = float((after != before).mean())
    sd = float(np.sqrt(max(flip - delta**2, 0) / n_public))
    # Rows for an 80% chance of seeing a true gain at two-sided 5%: (1.96 + 0.84)^2 * f / delta^2.
    needed = float(2.8**2 * flip / delta**2) if abs(delta) > 1e-9 else float("inf")
    return dict(d_blend=delta, flip=flip, sd_public=sd, z_public=delta / sd if sd else 0.0, rows_needed=needed)
