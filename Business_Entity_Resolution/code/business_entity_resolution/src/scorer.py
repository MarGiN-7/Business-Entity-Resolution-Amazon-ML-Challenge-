from typing import Dict, Set, Tuple


def fbeta_single(y_true: Set[str], y_pred: Set[str], beta: float = 0.5) -> float:
    if not y_true and not y_pred:
        return 1.0
    if not y_pred:
        return 0.0 if y_true else 1.0
    if not y_true:
        return 0.0
    tp = len(y_true & y_pred)
    if tp == 0:
        return 0.0
    precision = tp / len(y_pred)
    recall = tp / len(y_true)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def compute_fbeta(y_true: Dict[str, Set[str]],
                  y_pred: Dict[str, Set[str]],
                  beta: float = 0.5,
                  s1_subset: Set[str] = None) -> Tuple[float, Dict[str, float]]:
    if s1_subset is None:
        all_s1 = set(y_true.keys()) | set(y_pred.keys())
    else:
        all_s1 = s1_subset
    per_entity = {}
    total = 0.0
    count = 0
    for s1 in all_s1:
        gt = y_true.get(s1, set())
        pr = y_pred.get(s1, set())
        f = fbeta_single(gt, pr, beta)
        per_entity[s1] = f
        total += f
        count += 1
    macro = total / count if count else 0.0
    return macro, per_entity


def blocking_stats(y_true: Dict[str, Set[str]],
                   candidates: Dict[str, Set[str]],
                   s1_subset: Set[str] = None) -> Dict[str, float]:
    if s1_subset is None:
        all_s1 = set(y_true.keys())
    else:
        all_s1 = s1_subset
    total_gt = 0
    found_gt = 0
    total_cand = 0
    for s1 in all_s1:
        gt = y_true.get(s1, set())
        cand = candidates.get(s1, set())
        total_gt += len(gt)
        found_gt += len(gt & cand)
        total_cand += len(cand)
    recall_ceiling = found_gt / total_gt if total_gt else 1.0
    avg_cand = total_cand / len(all_s1) if all_s1 else 0
    return {
        "recall_ceiling": recall_ceiling,
        "avg_candidates_per_s1": avg_cand,
        "total_candidates": total_cand,
        "total_true_matches": total_gt,
        "found_true_matches": found_gt,
    }
