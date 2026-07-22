from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list[object]]


@dataclass(frozen=True)
class EnsembleDecision:
    selected_index: int
    selected_holding: np.ndarray
    votes: np.ndarray
    dispersion: float
    mode: str
    confidence_matrix: np.ndarray


def holding_dispersion(holdings: np.ndarray, eps: float = 1e-12) -> float:
    """Average min-max-normalized cross-agent holding standard deviation."""
    arr = np.asarray(holdings, dtype=float)
    if arr.ndim != 2:
        raise ValueError("holdings must be a 2D array shaped (agents, stocks)")
    sigma = arr.std(axis=0)
    sigma_range = sigma.max() - sigma.min()
    if sigma_range <= eps:
        return 0.0
    normalized = (sigma - sigma.min()) / (sigma_range + eps)
    return float(normalized.mean())


def classifier_specs(group: int, random_state: int = 42) -> list[ClassifierSpec]:
    svm_specs = [
        ClassifierSpec(
            name=f"svm_{kernel}",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel=kernel,
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid={
                "svc__C": [0.1, 1.0, 10.0],
                **(
                    {"svc__gamma": ["scale", "auto"]}
                    if kernel in {"rbf", "poly", "sigmoid"}
                    else {}
                ),
            },
        )
        for kernel in ("rbf", "linear", "poly", "sigmoid")
    ]
    logistic_specs = [
        ClassifierSpec(
            name=f"logistic_{penalty}",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            penalty=penalty,
                            solver="saga",
                            l1_ratio=0.5 if penalty == "elasticnet" else None,
                            max_iter=50000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid={
                "logistic__C": [0.1, 1.0, 10.0],
                **(
                    {"logistic__l1_ratio": [0.25, 0.5, 0.75]}
                    if penalty == "elasticnet"
                    else {}
                ),
            },
        )
        for penalty in ("l1", "l2", "elasticnet")
    ]
    tree_specs = [
        ClassifierSpec(
            name=f"decision_tree_{criterion}",
            estimator=DecisionTreeClassifier(
                criterion=criterion,
                random_state=random_state,
            ),
            param_grid={"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10]},
        )
        for criterion in ("gini", "entropy")
    ]

    groups = {
        1: svm_specs,
        2: logistic_specs,
        3: tree_specs,
        4: svm_specs + logistic_specs,
        5: svm_specs + logistic_specs + tree_specs,
    }
    if group not in groups:
        raise ValueError("classifier group must be one of 1, 2, 3, 4, or 5")
    return groups[group]


def train_classifier_group(
    holdings_by_agent: Iterable[np.ndarray],
    group: int,
    *,
    random_state: int = 42,
    grid_search: bool = False,
    cv: int = 5,
) -> list[tuple[str, BaseEstimator]]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for label, holdings in enumerate(holdings_by_agent):
        block = np.asarray(holdings, dtype=float)
        if block.ndim != 2:
            raise ValueError("each holdings block must be shaped (samples, stocks)")
        feature_blocks.append(block)
        label_blocks.append(np.full(block.shape[0], label, dtype=int))

    x = np.vstack(feature_blocks)
    y = np.concatenate(label_blocks)
    class_counts = np.bincount(y)
    usable_cv = int(min(cv, class_counts.min())) if class_counts.size else 0
    if usable_cv < 2:
        grid_search = False

    trained: list[tuple[str, BaseEstimator]] = []
    for spec in classifier_specs(group, random_state=random_state):
        estimator = clone(spec.estimator)
        if grid_search:
            estimator = GridSearchCV(
                estimator=estimator,
                param_grid=spec.param_grid,
                cv=usable_cv,
                n_jobs=1,
            )
        estimator.fit(x, y)
        trained.append((spec.name, estimator))
    return trained


def confidence_matrix(
    classifiers: Iterable[tuple[str, BaseEstimator] | BaseEstimator],
    candidate_holdings: np.ndarray,
    true_agent_labels: Iterable[int],
) -> np.ndarray:
    candidates = np.asarray(candidate_holdings, dtype=float)
    labels = list(true_agent_labels)
    rows: list[list[float]] = []
    for item in classifiers:
        estimator = item[1] if isinstance(item, tuple) else item
        probabilities = estimator.predict_proba(candidates)
        classes = np.asarray(getattr(estimator, "classes_", []))
        if classes.size == 0 and hasattr(estimator, "best_estimator_"):
            classes = np.asarray(estimator.best_estimator_.classes_)
        row = []
        for candidate_index, label in enumerate(labels):
            matching = np.where(classes == label)[0]
            row.append(float(probabilities[candidate_index, matching[0]]) if matching.size else 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def select_holding_from_confidence(
    candidate_holdings: np.ndarray,
    q_matrix: np.ndarray,
    tau: float,
    *,
    dispersion: float | None = None,
) -> EnsembleDecision:
    candidates = np.asarray(candidate_holdings, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    if q.ndim != 2 or q.shape[1] != candidates.shape[0]:
        raise ValueError("q_matrix must be shaped (classifiers, candidate_holdings)")

    sigma_bar = holding_dispersion(candidates) if dispersion is None else float(dispersion)
    low_variance = sigma_bar < tau
    vote_indices = np.argmax(q, axis=1) if low_variance else np.argmin(q, axis=1)
    votes = np.bincount(vote_indices, minlength=candidates.shape[0])
    tied = np.flatnonzero(votes == votes.max())
    if tied.size == 1:
        selected = int(tied[0])
    else:
        tie_scores = q[:, tied].mean(axis=0)
        selected = int(tied[np.argmax(tie_scores) if low_variance else np.argmin(tie_scores)])

    return EnsembleDecision(
        selected_index=selected,
        selected_holding=candidates[selected].copy(),
        votes=votes,
        dispersion=sigma_bar,
        mode="aggressive" if low_variance else "conservative",
        confidence_matrix=q.copy(),
    )


def select_holding(
    classifiers: Iterable[tuple[str, BaseEstimator] | BaseEstimator],
    candidate_holdings: np.ndarray,
    true_agent_labels: Iterable[int],
    tau: float,
) -> EnsembleDecision:
    q = confidence_matrix(classifiers, candidate_holdings, true_agent_labels)
    return select_holding_from_confidence(candidate_holdings, q, tau)


def tau_grid(start: float = 0.01, stop: float = 0.89, step: float = 0.01) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(count), 10)
