from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=500,
                    l2_regularization=0.2,
                    max_leaf_nodes=63,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            ),
        ]
    )
