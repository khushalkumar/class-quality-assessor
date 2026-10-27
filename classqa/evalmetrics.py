"""Offline evaluation metrics (pure Python, no API calls): confusion matrix, quadratic weighted kappa,
threshold fitting and leave-one-out, rank correlation. Used by eval/advanced_eval.py and the CI regression gate."""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from . import rubric

R = rubric.RATINGS  # ordinal: Poor < Average < Good < Excellent


def confusion(actual: Sequence[str], pred: Sequence[str], labels: Sequence[str] = R) -> list[list[int]]:
    """rows = actual, cols = predicted."""
    m = [[0] * len(labels) for _ in labels]
    for a, p in zip(actual, pred, strict=True):
        m[labels.index(a)][labels.index(p)] += 1
    return m


def quadratic_kappa(actual: Sequence[str], pred: Sequence[str], labels: Sequence[str] = R) -> float:
    """Cohen's kappa with quadratic weights over the ordinal label set (1 = perfect, 0 = chance)."""
    k, n = len(labels), len(actual)
    if n == 0:
        return float("nan")
    o = confusion(actual, pred, labels)
    ra = [sum(row) for row in o]
    cp = [sum(o[i][j] for i in range(k)) for j in range(k)]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * o[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * ra[i] * cp[j] / n for i in range(k) for j in range(k))
    return 1.0 if den == 0 else 1 - num / den


def exact(actual, pred) -> int:
    return sum(a == p for a, p in zip(actual, pred, strict=True))


def within_one(actual, pred) -> int:
    return sum(abs(R.index(a) - R.index(p)) <= 1 for a, p in zip(actual, pred, strict=True))


def mae_levels(actual, pred) -> float:
    return sum(abs(R.index(a) - R.index(p)) for a, p in zip(actual, pred, strict=True)) / len(actual)


def label(score: float, cuts: Sequence[float], capped: bool = False) -> str:
    """cuts = (Average, Good, Excellent) lower bounds. `capped` applies the major-red-flag cap."""
    lvl = sum(score >= c for c in cuts)
    if capped:
        lvl = min(lvl, R.index(rubric.MAJOR_FLAG_CAP))
    return R[lvl]


def current_cuts() -> tuple[float, float, float]:
    th = {name: t for t, name in rubric.THRESHOLDS}
    return th["Average"], th["Good"], th["Excellent"]


def fit_cuts(rows: Sequence[dict], prior: Sequence[float] | None = None, step: float = 0.05):
    """Grid-search the three cut points on `rows` ({score, capped, actual}). Objective: max exact, then max
    within-one, then the smallest total move from `prior` (the hand-set thresholds) - so ties stay conservative."""
    prior = tuple(prior or current_cuts())
    grid = [round(1 + i * step, 2) for i in range(int(4 / step) + 1)]
    act = [r["actual"] for r in rows]
    best, best_key = prior, None
    for cuts in combinations(grid, 3):
        pred = [label(r["score"], cuts, r["capped"]) for r in rows]
        key = (exact(act, pred), within_one(act, pred), -round(sum(abs(a - b) for a, b in zip(cuts, prior, strict=False)), 4))
        if best_key is None or key > best_key:
            best, best_key = cuts, key
    return best


def leave_one_out(rows: Sequence[dict], prior: Sequence[float] | None = None) -> list[str]:
    """Prediction for each row with thresholds fitted on all other rows."""
    out = []
    for i, r in enumerate(rows):
        cuts = fit_cuts([x for j, x in enumerate(rows) if j != i], prior)
        out.append(label(r["score"], cuts, r["capped"]))
    return out


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return float("nan") if vx == 0 or vy == 0 else cov / (vx * vy)


def summary(rows: Sequence[dict]) -> dict:
    """In-sample (current thresholds) and leave-one-out metrics for rows {id, actual, score, capped}."""
    act = [r["actual"] for r in rows]
    ins = [label(r["score"], current_cuts(), r["capped"]) for r in rows]
    loo = leave_one_out(rows)
    n = len(rows)
    return {
        "n": n,
        "in_sample": {"pred": ins, "exact": exact(act, ins), "within_one": within_one(act, ins),
                      "qwk": round(quadratic_kappa(act, ins), 3), "mae_levels": round(mae_levels(act, ins), 2),
                      "confusion": confusion(act, ins)},
        "leave_one_out": {"pred": loo, "exact": exact(act, loo), "within_one": within_one(act, loo),
                          "qwk": round(quadratic_kappa(act, loo), 3), "mae_levels": round(mae_levels(act, loo), 2),
                          "confusion": confusion(act, loo)},
    }
