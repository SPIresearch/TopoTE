from __future__ import annotations

import math
from typing import Iterable


def kendall_tau_b(x: Iterable[float], y: Iterable[float]) -> float:
    """Kendall's tau-b for rankings with ties.

    输入是两组等长的数值（分数越大越好）。
    输出范围 [-1, 1]，越大表示排序越一致。
    """

    xs = list(x)
    ys = list(y)
    n = min(len(xs), len(ys))
    xs = xs[:n]
    ys = ys[:n]
    if n < 2:
        return float("nan")

    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0

    for i in range(n - 1):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            dx = xi - xs[j]
            dy = yi - ys[j]

            sx = 0
            if dx > 0:
                sx = 1
            elif dx < 0:
                sx = -1

            sy = 0
            if dy > 0:
                sy = 1
            elif dy < 0:
                sy = -1

            if sx == 0 and sy == 0:
                # pair tied in both
                continue
            if sx == 0:
                tie_x += 1
                continue
            if sy == 0:
                tie_y += 1
                continue
            if sx == sy:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom
