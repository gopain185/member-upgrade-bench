"""The grid: what happens to a blend when one member gets better, or a new member joins.

    python -m upgradebench.run            # writes results/replace.csv and results/add.csv
    python -m upgradebench.run --quick    # small grid for a smoke test
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .core import blend, correct, detection, member, noise_of_mean, one_hot, signal_for

ROOT = Path(__file__).resolve().parent.parent
N_ROWS = 36_000
SEEDS = 4
STRONG = 0.60   # solo accuracy of the other members
BEFORE = 0.52   # solo accuracy of the member that gets replaced
K = [2, 4, 8]
RHO = [0.3, 0.5, 0.7, 0.9]
RHO_T = {"same": 1.0, "lower": 0.5}   # the target member's rho, relative to the rest of the blend
SHARE = [0.1, 0.2, 0.33, 0.5]
GAIN = [0.01, 0.02, 0.044, 0.08]
SOLO_ADD = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
KEEP = 0.5    # share of the old member's private noise variance a retrained member keeps
WEIGHTS = np.round(np.arange(0.0, 0.81, 0.05), 2)   # searched for the best share


def _streams(k: int, n: int, classes: int, seed: int):
    rng = np.random.default_rng(seed)
    shape = (n, classes)
    shared = rng.standard_normal(shape, dtype=np.float32)
    private = [rng.standard_normal(shape, dtype=np.float32) for _ in range(k)]
    old = rng.standard_normal(shape, dtype=np.float32)
    fresh = rng.standard_normal(shape, dtype=np.float32)
    return shared, private, old, fresh


def _rest(k, rho, onehot, shared, private, strong_signal):
    """The mean of the k strong members, scaled to unit noise so that `share` is a share of comparable scores."""
    mean = sum(member(strong_signal, rho, onehot, shared, p) for p in private) / k
    return mean / noise_of_mean(k, rho)


def _best_share(rest, target) -> tuple[float, float]:
    scores = [correct(blend(rest, target, w)).mean() for w in WEIGHTS]
    i = int(np.argmax(scores))
    return float(WEIGHTS[i]), float(scores[i])


def replace_grid(quick: bool = False, n: int = N_ROWS, seeds: int = SEEDS) -> pd.DataFrame:
    """One member of solo accuracy BEFORE is retrained to BEFORE + gain and keeps its weight in the blend."""
    rows = []
    strong, old_a = signal_for(STRONG), signal_for(BEFORE)
    gains = {g: signal_for(BEFORE + g) for g in (GAIN[1:3] if quick else GAIN)}
    onehot = one_hot(n)
    for seed, k, rho in itertools.product(range(1 if quick else seeds), K[:2] if quick else K, RHO[1:2] if quick else RHO):
        shared, private, old_noise, fresh = _streams(k, n, onehot.shape[1], 1000 + seed)
        rest = _rest(k, rho, onehot, shared, private, strong)
        alone = correct(rest)
        for rel, mult in RHO_T.items():
            rho_t = rho * mult
            old = member(old_a, rho_t, onehot, shared, old_noise)
            new_noise = np.sqrt(KEEP) * old_noise + np.sqrt(1 - KEEP) * fresh
            solo_old = correct(old)
            best_old_w, best_old = _best_share(rest, old)
            for gain, a in gains.items():
                new = member(a, rho_t, onehot, shared, new_noise)
                solo_new = correct(new)
                d_solo = float(solo_new.mean() - solo_old.mean())
                best_new_w, best_new = _best_share(rest, new)
                for share in SHARE:
                    before, after = correct(blend(rest, old, share)), correct(blend(rest, new, share))
                    rows.append(dict(seed=seed, k=k, rho=rho, rho_t=rel, share=share, gain=gain, d_solo=d_solo,
                                     solo_flip=float((solo_new != solo_old).mean()), rest_acc=float(alone.mean()),
                                     blend_before=float(before.mean()), blend_after=float(after.mean()),
                                     retention=float(after.mean() - before.mean()) / d_solo,
                                     best_share_old=best_old_w, best_share_new=best_new_w,
                                     d_best=best_new - best_old, **detection(before, after)))
    return pd.DataFrame(rows)


def add_grid(quick: bool = False, n: int = N_ROWS, seeds: int = SEEDS) -> pd.DataFrame:
    """A new member joins a blend that did not have it, against the blend alone."""
    rows = []
    strong = signal_for(STRONG)
    solos = {s: signal_for(s) for s in (SOLO_ADD[1:3] if quick else SOLO_ADD)}
    onehot = one_hot(n)
    for seed, k, rho in itertools.product(range(1 if quick else seeds), K[:2] if quick else K, RHO[1:2] if quick else RHO):
        shared, private, new_noise, _ = _streams(k, n, onehot.shape[1], 2000 + seed)
        rest = _rest(k, rho, onehot, shared, private, strong)
        alone = correct(rest)
        for (rel, mult), (solo, a) in itertools.product(RHO_T.items(), solos.items()):
            new = member(a, rho * mult, onehot, shared, new_noise)
            best_w, best = _best_share(rest, new)
            for share in SHARE:
                after = correct(blend(rest, new, share))
                rows.append(dict(seed=seed, k=k, rho=rho, rho_t=rel, share=share, solo=solo, rest_acc=float(alone.mean()),
                                 blend_after=float(after.mean()), best_share=best_w, d_best=best - float(alone.mean()),
                                 **detection(alone, after)))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Means over seeds, plus the seed-to-seed sd of the blend change."""
    numeric = [c for c in df.columns if c not in keys + ["seed"]]
    out = df.groupby(keys)[numeric].mean(numeric_only=True).reset_index()
    out["d_blend_sd"] = df.groupby(keys)["d_blend"].std().to_numpy() if df.seed.nunique() > 1 else np.nan
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    suffix = "_quick" if a.quick else ""
    rep = summarise(replace_grid(a.quick), ["k", "rho", "rho_t", "share", "gain"])
    add = summarise(add_grid(a.quick), ["k", "rho", "rho_t", "share", "solo"])
    rep.to_csv(out / f"replace{suffix}.csv", index=False, float_format="%.6g")
    add.to_csv(out / f"add{suffix}.csv", index=False, float_format="%.6g")
    print(len(rep), "replace cells,", len(add), "add cells")
