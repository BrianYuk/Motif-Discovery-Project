import numpy as np

from .alphabet import revcomp_pwm


def information_content(pwm: np.ndarray, background=(0.25, 0.25, 0.25, 0.25)) -> float:
    # IC = sum_j sum_b p_bj log2(p_bj / q_b), in bits.
    p = np.clip(np.asarray(pwm, dtype=float), 1e-12, 1.0)
    q = np.asarray(background, dtype=float)[:, None]
    return float((p * np.log2(p / q)).sum())


def _column_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Pearson correlation between matching columns; 0 where a column is flat.
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 1e-12, num / den, 0.0)
    return r


def pwm_similarity(found: np.ndarray, planted: np.ndarray, min_overlap: int = None) -> dict:
    # Tomtom-style score between two PWMs (4 x W each, widths may differ).
    found = np.asarray(found, dtype=float)
    planted = np.asarray(planted, dtype=float)
    wf, wp = found.shape[1], planted.shape[1]
    if min_overlap is None:
        min_overlap = max(1, min(wf, wp) // 2)
    best = {"score": -np.inf, "offset": 0, "strand": "+"}
    for strand, f in (("+", found), ("-", revcomp_pwm(found))):
        for offset in range(-(wf - min_overlap), wp - min_overlap + 1):
            lo_p, hi_p = max(0, offset), min(wp, offset + wf)
            if hi_p - lo_p < min_overlap:
                continue
            cols_p = planted[:, lo_p:hi_p]
            cols_f = f[:, lo_p - offset:hi_p - offset]
            score = float(_column_corr(cols_f, cols_p).sum() / max(wf, wp))
            if score > best["score"]:
                best = {"score": score, "offset": offset, "strand": strand}
    return best


def site_metrics(predicted, true, tolerance: int = 2) -> dict:
    # Precision, recall and F1 of predicted site starts.
    if len(predicted) != len(true):
        raise ValueError("predicted and true must have one entry per sequence")
    tp = fp = fn = 0
    for p, t in zip(predicted, true):
        if p is not None and t is not None and abs(p - t) <= tolerance:
            tp += 1
        else:
            if p is not None:
                fp += 1
            if t is not None:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def align_sites(predicted, offset: int, strand: str = "+"):
    if strand != "+":
        return list(predicted)
    return [None if a is None else a - offset for a in predicted]


def consensus(pwm: np.ndarray) -> str:
    """IUPAC-free consensus: the most likely base per column, lower case if weak (<0.5)."""
    out = []
    for col in np.asarray(pwm).T:
        b = "ACGT"[int(np.argmax(col))]
        out.append(b if col.max() >= 0.5 else b.lower())
    return "".join(out)
