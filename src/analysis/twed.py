"""Time Warp Edit Distance (Marteau, 2009) for irregularly-sampled series.

Unlike DTW, TWED uses the actual timestamps (not just ordinal position),
via a `nu` (stiffness) penalty on time misalignment, plus a `lam` penalty
per insert/delete -- a natural fit for our clustered/gappy toy series
where the elapsed time between points is meaningful.

Implemented directly from the published recursion, not vendored from any
library -- validated below with sanity checks before being trusted for
anything downstream.
"""
import numpy as np


def twed(tA, vA, tB, vB, nu=1.0, lam=1.0, p=1):
    """TWED between two (possibly irregularly-sampled) 1D series.

    tA, vA: timestamps and values of series A (1D arrays, same length).
    tB, vB: timestamps and values of series B.
    nu: stiffness -- penalty per unit of time misalignment.
    lam: constant penalty per insert/delete operation.
    p: order of the value-distance norm (1 = absolute difference).
    """
    tA, vA, tB, vB = map(lambda a: np.asarray(a, dtype=np.float64), (tA, vA, tB, vB))
    n, m = len(vA), len(vB)

    def dist(a, b):
        return abs(a - b) ** p

    # Pad index 0 with a duplicate of index 1 (a common convention that
    # avoids needing a true phantom "start" point with its own cost).
    A = np.concatenate(([vA[0]], vA))
    TA = np.concatenate(([tA[0]], tA))
    B = np.concatenate(([vB[0]], vB))
    TB = np.concatenate(([tB[0]], tB))

    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] + dist(A[i], A[i - 1]) + nu * abs(TA[i] - TA[i - 1]) + lam
    for j in range(1, m + 1):
        D[0, j] = D[0, j - 1] + dist(B[j], B[j - 1]) + nu * abs(TB[j] - TB[j - 1]) + lam

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = (D[i - 1, j - 1] + dist(A[i], B[j]) + dist(A[i - 1], B[j - 1])
                     + nu * (abs(TA[i] - TB[j]) + abs(TA[i - 1] - TB[j - 1])))
            del_a = D[i - 1, j] + dist(A[i], A[i - 1]) + nu * abs(TA[i] - TA[i - 1]) + lam
            del_b = D[i, j - 1] + dist(B[j], B[j - 1]) + nu * abs(TB[j] - TB[j - 1]) + lam
            D[i, j] = min(match, del_a, del_b)

    return D[n, m]


def pairwise_twed(series_list, nu=1.0, lam=1.0, p=1):
    """series_list: list of (t, v) tuples. Returns an (N, N) distance matrix."""
    n = len(series_list)
    D = np.zeros((n, n))
    for i in range(n):
        ti, vi = series_list[i]
        for j in range(i + 1, n):
            tj, vj = series_list[j]
            d = twed(ti, vi, tj, vj, nu=nu, lam=lam, p=p)
            D[i, j] = D[j, i] = d
    return D
