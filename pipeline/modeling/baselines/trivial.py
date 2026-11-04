"""
Trivial baselines: random / subtype-only / pixel-mean.

- RandomBaseline      : predict positive with fixed probability
- SubtypeOnlyBaseline : predict ER from PAM50 subtype label (controls for ER-PAM50 correlation)
- MeanEmbedBaseline   : logistic regression on mean-pooled tile embedding (pixel-mean proxy)
"""

import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


class RandomBaseline:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.pos_rate = 0.5

    def fit(self, X: list[np.ndarray], y: np.ndarray, subtypes: list = None):
        self.pos_rate = y.mean()

    def predict_proba(self, X: list[np.ndarray], subtypes: list = None) -> np.ndarray:
        return self.rng.uniform(0, 1, len(X)).astype(np.float32)


class SubtypeOnlyBaseline:
    """Predicts ER status from PAM50 subtype label only (no embedding used).
    Learns P(ER+|PAM50=subtype) from training set.
    """

    def __init__(self):
        self.subtype_probs = {}
        self.default_prob = 0.5

    def fit(self, X: list[np.ndarray], y: np.ndarray, subtypes: list):
        groups = defaultdict(list)
        for subtype, label in zip(subtypes, y):
            groups[subtype].append(label)
        for subtype, labels in groups.items():
            self.subtype_probs[subtype] = float(np.mean(labels))
        self.default_prob = float(y.mean())

    def predict_proba(self, X: list[np.ndarray], subtypes: list) -> np.ndarray:
        return np.array(
            [self.subtype_probs.get(s, self.default_prob) for s in subtypes],
            dtype=np.float32,
        )


class MeanEmbedBaseline:
    """Logistic regression on mean-pooled tile embeddings (pixel-mean proxy)."""

    def __init__(self, seed: int = 42):
        self.clf = LogisticRegression(max_iter=1000, random_state=seed)

    def _pool(self, X: list[np.ndarray]) -> np.ndarray:
        return np.stack([x.mean(axis=0) for x in X])

    def fit(self, X: list[np.ndarray], y: np.ndarray, subtypes: list = None):
        if len(np.unique(y)) < 2:
            self.clf = None
            self._fallback_prob = float(y.mean())
            return
        self.clf.fit(self._pool(X), y)

    def predict_proba(self, X: list[np.ndarray], subtypes: list = None) -> np.ndarray:
        if self.clf is None:
            return np.full(len(X), getattr(self, '_fallback_prob', 0.5), dtype=np.float32)
        return self.clf.predict_proba(self._pool(X))[:, 1].astype(np.float32)


class MulticlassRandomBaseline:
    """Random baseline for multi-class (e.g. PAM50). Predicts training class distribution."""

    def __init__(self, seed: int = 42, num_classes: int = 5):
        self.rng = np.random.default_rng(seed)
        self.num_classes = num_classes

    def fit(self, X: list[np.ndarray], y: np.ndarray, subtypes: list = None):
        pass

    def predict_proba(self, X: list[np.ndarray], subtypes: list = None) -> np.ndarray:
        raw = self.rng.uniform(0, 1, (len(X), self.num_classes))
        return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)


class MulticlassMajorityBaseline:
    """Always predicts the majority class distribution from train set."""

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.probs = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)

    def fit(self, X: list[np.ndarray], y: np.ndarray, subtypes: list = None):
        counts = np.bincount(y.astype(int), minlength=self.num_classes)
        majority = int(np.argmax(counts))
        self.probs = np.zeros(self.num_classes, dtype=np.float32)
        self.probs[majority] = 1.0

    def predict_proba(self, X: list[np.ndarray], subtypes: list = None) -> np.ndarray:
        return np.tile(self.probs, (len(X), 1))


class MulticlassMeanEmbedBaseline:
    """Multinomial logistic regression on mean-pooled tile embeddings."""

    def __init__(self, seed: int = 42, num_classes: int = 5):
        self.clf = LogisticRegression(max_iter=1000, random_state=seed, multi_class="multinomial")
        self.num_classes = num_classes

    def _pool(self, X: list[np.ndarray]) -> np.ndarray:
        return np.stack([x.mean(axis=0) for x in X])

    def fit(self, X: list[np.ndarray], y: np.ndarray, subtypes: list = None):
        self.clf.fit(self._pool(X), y.astype(int))

    def predict_proba(self, X: list[np.ndarray], subtypes: list = None) -> np.ndarray:
        pooled = self._pool(X)
        proba = np.zeros((len(X), self.num_classes), dtype=np.float32)
        classes = self.clf.classes_
        raw = self.clf.predict_proba(pooled)
        for i, c in enumerate(classes):
            proba[:, c] = raw[:, i]
        return proba


def bootstrap_auc_ci_multiclass(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_classes: int,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Bootstrap 95% CI for macro OvR AUC (multi-class)."""
    from sklearn.preprocessing import label_binarize
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, ys, multi_class="ovr", average="macro"))
        except ValueError:
            continue
    if not aucs:
        return None, None
    alpha = (1 - ci) / 2
    return (
        round(float(np.percentile(aucs, alpha * 100)), 4),
        round(float(np.percentile(aucs, (1 - alpha) * 100)), 4),
    )


def evaluate_multiclass(name: str, proba: np.ndarray, y: np.ndarray, num_classes: int, add_ci: bool = False) -> dict:
    from sklearn.preprocessing import label_binarize
    preds = proba.argmax(axis=1)
    has_all_classes = len(np.unique(y)) == num_classes
    auc = auprc = None
    if has_all_classes:
        try:
            auc = round(float(roc_auc_score(y, proba, multi_class="ovr", average="macro")), 4)
            y_bin = label_binarize(y, classes=list(range(num_classes)))
            auprc = round(float(average_precision_score(y_bin, proba, average="macro")), 4)
        except ValueError:
            pass
    metrics = {
        "baseline": name,
        "auc": auc,
        "auprc": auprc,
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
        "n": len(y),
    }
    if add_ci and auc is not None:
        lo, hi = bootstrap_auc_ci_multiclass(y, proba, num_classes)
        if lo is not None:
            metrics["auc_ci_95"] = [lo, hi]
    return metrics


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Bootstrap 95% CI for AUC. Returns (lower, upper)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    alpha = (1 - ci) / 2
    return (
        round(float(np.percentile(aucs, alpha * 100)), 4),
        round(float(np.percentile(aucs, (1 - alpha) * 100)), 4),
    )


def evaluate(name: str, proba: np.ndarray, y: np.ndarray, add_ci: bool = False) -> dict:
    preds = (proba >= 0.5).astype(int)
    has_both = len(np.unique(y)) > 1
    auc   = round(float(roc_auc_score(y, proba)), 4) if has_both else None
    auprc = round(float(average_precision_score(y, proba)), 4) if has_both else None
    metrics = {
        "baseline": name,
        "auc": auc,
        "auprc": auprc,
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
        "n": len(y),
    }
    if add_ci and auc is not None:
        lo, hi = bootstrap_auc_ci(y, proba)
        metrics["auc_ci_95"] = [lo, hi]
    return metrics
