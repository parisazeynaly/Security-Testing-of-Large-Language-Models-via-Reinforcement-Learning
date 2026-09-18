"""
CausalBreakerSCM: a linear Structural Causal Model used for SCM-informed
intermediate feedback and causal-graph-guided action selection.

The graph structure encodes the structural assumptions used by the research
pipeline. Runtime outcome weights are updated from observed factor/outcome
pairs using OLS. Consequently, predictions and simulated interventions from
this module are model-based structural guidance under the fitted SCM; they are
not experimentally identified causal-effect estimates.
"""

import threading
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import IDX, OBS_KEYS


class CausalBreakerSCM:
    def __init__(self, sigma: float = 0.01):
        self.n = len(OBS_KEYS)
        self.adj_matrix = np.zeros((self.n, self.n), dtype=np.float32)

        # Structural graph used by the research pipeline
        self.adj_matrix[IDX["instructional_style"], IDX["responsibility_externalization"]] = 0.35
        self.adj_matrix[IDX["instructional_style"], IDX["obfuscation_techniques"]] = 0.30
        self.adj_matrix[IDX["instructional_style"], IDX["hypothetical_framing"]] = 0.20

        self.adj_matrix[IDX["responsibility_externalization"], IDX["obfuscation_techniques"]] = 0.30
        self.adj_matrix[IDX["responsibility_externalization"], IDX["hypothetical_framing"]] = 0.25

        self.adj_matrix[IDX["obfuscation_techniques"], IDX["hypothetical_framing"]] = 0.40

        self.adj_matrix[IDX["malicious_intent"], IDX["imperative_tone"]] = 0.35

        self.adj_matrix[IDX["imperative_tone"], IDX["responsibility_externalization"]] = 0.20
        self.adj_matrix[IDX["imperative_tone"], IDX["obfuscation_techniques"]] = 0.25
        self.adj_matrix[IDX["imperative_tone"], IDX["hypothetical_framing"]] = 0.35

        self.y_weights = np.array([
            0.10,  # instructional_style
            0.10,  # responsibility_externalization
            0.25,  # obfuscation_techniques
            0.30,  # hypothetical_framing
            0.20,  # imperative_tone
            0.05,  # malicious_intent
        ], dtype=np.float32)

        self.sigma = sigma
        self._train_mode = True
        self._lock = threading.Lock()

    def set_train_mode(self):
        self._train_mode = True

    def set_eval_mode(self):
        self._train_mode = False

    def _structural_equations(self, x_init, adj, add_noise=False):
        x_hat = np.clip(x_init.copy(), 0.0, 1.0).astype(np.float32)

        topo_order = [
            IDX["instructional_style"],
            IDX["malicious_intent"],
            IDX["imperative_tone"],
            IDX["responsibility_externalization"],
            IDX["obfuscation_techniques"],
            IDX["hypothetical_framing"],
        ]

        for j in topo_order:
            if not np.any(adj[:, j] != 0.0):
                continue

            value = float(np.dot(adj[:, j], x_hat))

            if add_noise and self._train_mode:
                value += np.random.normal(0.0, self.sigma)

            x_hat[j] = float(np.clip(value, 0.0, 1.0))

        return x_hat

    def predict(self, obs, add_noise=False) -> float:
        obs = np.clip(obs, 0.0, 1.0).astype(np.float32)

        with self._lock:
            adj = self.adj_matrix.copy()
            yw = self.y_weights.copy()

        x_hat = self._structural_equations(obs, adj, add_noise=add_noise)
        return float(np.clip(np.dot(yw, x_hat), 0.0, 1.0))

    def intervene(self, obs, factor_name: str, value: float = 1.0) -> float:
        obs = np.clip(obs, 0.0, 1.0).astype(np.float32)

        if factor_name not in IDX:
            raise ValueError(f"Unknown factor: {factor_name}")

        target_idx = IDX[factor_name]

        with self._lock:
            local_adj = self.adj_matrix.copy()
            yw = self.y_weights.copy()

        x_do = obs.copy()
        x_do[target_idx] = float(np.clip(value, 0.0, 1.0))

        # do-intervention: remove incoming edges into the intervened node
        local_adj[:, target_idx] = 0.0

        x_hat = self._structural_equations(x_do, local_adj, add_noise=False)
        return float(np.clip(np.dot(yw, x_hat), 0.0, 1.0))

    def update(self, buffer: List[Dict]) -> bool:
        """Re-estimate outcome weights via OLS on observed factor/outcome pairs."""
        if len(buffer) < 100:
            return False

        df = pd.DataFrame(buffer).dropna()

        required = OBS_KEYS + ["y"]
        if not all(c in df.columns for c in required):
            return False

        if df["y"].std() < 0.01:
            print("Skipped SCM y_weight update: insufficient outcome variation.")
            return False

        try:
            X = df[OBS_KEYS].values.astype(float)
            y = df["y"].values.astype(float)

            X_aug = np.column_stack([X, np.ones(len(X))])
            coef = np.linalg.lstsq(X_aug, y, rcond=None)[0][:-1]

            coef = np.abs(coef) + 1e-6
            coef = coef / coef.sum()

            with self._lock:
                self.y_weights = coef.astype(np.float32)

            return True

        except Exception as e:
            print(f"SCM update failed: {e}")
            return False

    def summary(self) -> str:
        with self._lock:
            yw = self.y_weights.copy()

        lines = [
            "=" * 60,
            "CausalBreakerSCM — Structural Guidance Model",
            "=" * 60,
        ]
        for factor, i in IDX.items():
            lines.append(f"{factor:35s} -> Y weight: {yw[i]:.3f}")
        lines.append("-" * 60)
        lines.append(f"Mode: {'train' if self._train_mode else 'eval'}")
        lines.append("=" * 60)

        return "\n".join(lines)


# Module-level singleton, matching the original notebook's global SCM usage.
SCM = CausalBreakerSCM()
