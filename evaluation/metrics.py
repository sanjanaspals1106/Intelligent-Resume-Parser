"""Plain-numpy implementations of the extraction and ranking metrics used by the evaluation."""
from __future__ import annotations

import math
import re


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def items_match(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a == b or (min(len(a), len(b)) >= 4 and (a in b or b in a)))


def prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    r = tp / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "predicted": n_pred, "gold": n_gt}


def count_matches(pred: list, gold: list) -> int:
    """Greedy one-to-one matching of predicted to gold items."""
    used, tp = set(), 0
    for p in pred:
        for i, g in enumerate(gold):
            if i not in used and items_match(p, g):
                used.add(i)
                tp += 1
                break
    return tp


def dcg(gains):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(ranked_grades: list, k: int) -> float:
    ideal = dcg(sorted(ranked_grades, reverse=True)[:k])
    return dcg(ranked_grades[:k]) / ideal if ideal else 0.0


def average_precision(ranked_grades: list, thr=2) -> float:
    hits, s = 0, 0.0
    for i, g in enumerate(ranked_grades, 1):
        if g >= thr:
            hits += 1
            s += hits / i
    return s / hits if hits else 0.0


def precision_at(ranked_grades: list, k: int, thr=2) -> float:
    return sum(g >= thr for g in ranked_grades[:k]) / k


def mrr(ranked_grades: list, thr=2) -> float:
    for i, g in enumerate(ranked_grades, 1):
        if g >= thr:
            return 1 / i
    return 0.0


def _ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def ranking_metrics(scores: list, grades: list, tiebreak: list) -> dict:
    order = sorted(range(len(scores)), key=lambda i: (-round(scores[i], 6), tiebreak[i]))
    rg = [grades[i] for i in order]
    return {"ndcg@5": ndcg_at(rg, 5), "ndcg@10": ndcg_at(rg, 10), "map": average_precision(rg), "p@5": precision_at(rg, 5), "mrr": mrr(rg), "spearman": spearman(scores, grades)}
