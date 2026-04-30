"""
FlowGuard — Anomaly Detection Module
Isolation Forest, LOF, XGBoost, and Autoencoder-based detection.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
)

from src.utils.helpers import load_config

logger = logging.getLogger("flowguard.anomaly")
warnings.filterwarnings("ignore")


# Feature Columns

FEATURE_COLS = [
    "tx_count", "mean_inter_tx_interval", "std_inter_tx_interval",
    "max_burst_rate", "unique_fn_count", "failed_tx_ratio",
    "mean_gas_ratio", "tx_value_entropy",
    "unique_states_visited", "state_transition_freq",
    "most_freq_transition_ratio", "reverse_transition_count",
    "terminal_state_reach_count", "mean_time_in_state",
    "state_sequence_entropy", "repeated_state_visit_count",
    "distinct_roles_assumed", "role_change_freq",
    "admin_fn_call_ratio", "guard_failure_rate",
    "delegatecall_count", "contract_creation_count",
    "unique_interacted_contracts", "self_ref_call_count",
    "total_value_transferred", "mean_value_per_tx",
    "value_variance", "max_single_tx_value",
    "cumulative_balance_delta", "value_gini",
    "token_transfer_count", "flash_loan_indicator",
]


# Isolation Forest

class IsolationForestDetector:

    def __init__(self, cfg: Dict = None):
        cfg = cfg or load_config()
        params = cfg["behavioral"]["anomaly_models"]["isolation_forest"]
        self.model = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = MinMaxScaler()
        self.weight = params["ensemble_weight"]
        self.fitted = False

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1] where 1 = most anomalous."""
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        raw = self.model.decision_function(X_scaled)
        # Invert and normalize: more negative = more anomalous
        scores = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction: 1 = anomalous, 0 = normal."""
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return (preds == -1).astype(int)


# Local Outlier Factor

class LOFDetector:

    def __init__(self, cfg: Dict = None):
        cfg = cfg or load_config()
        params = cfg["behavioral"]["anomaly_models"]["lof"]
        self.n_neighbors = params["n_neighbors"]
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.X_train_scaled = None

    def fit(self, X: np.ndarray):
        self.X_train_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute LOF scores for new data."""
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        # LOF with novelty=True for scoring new data
        lof = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(self.X_train_scaled) - 1),
            novelty=True,
        )
        lof.fit(self.X_train_scaled)
        raw = lof.decision_function(X_scaled)
        scores = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
        return scores


#  XGBoost Supervised Classifier

class XGBoostDetector:

    def __init__(self, cfg: Dict = None):
        cfg = cfg or load_config()
        params = cfg["behavioral"]["anomaly_models"]["xgboost"]
        self.weight = params["ensemble_weight"]
        self.scaler = MinMaxScaler()
        self.fitted = False

        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            self.available = True
        except ImportError:
            logger.warning("XGBoost not available — using sklearn GradientBoosting")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                random_state=42,
            )
            self.available = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return probability of being anomalous."""
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return probs


#  Autoencoder

class AutoencoderDetector:

    def __init__(self, cfg: Dict = None):
        cfg = cfg or load_config()
        params = cfg["behavioral"]["anomaly_models"]["autoencoder"]
        self.hidden_dims = params["hidden_dims"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.weight = params["ensemble_weight"]
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.model = None
        self.threshold = None

    def _build_model(self, input_dim: int):
        """Build PyTorch autoencoder if available, else numpy fallback."""
        try:
            import torch
            import torch.nn as nn

            class AE(nn.Module):
                def __init__(self, dims):
                    super().__init__()
                    layers_enc = []
                    prev = input_dim
                    for d in dims[:len(dims) // 2 + 1]:
                        layers_enc.append(nn.Linear(prev, d))
                        layers_enc.append(nn.ReLU())
                        prev = d
                    self.encoder = nn.Sequential(*layers_enc)

                    layers_dec = []
                    for d in dims[len(dims) // 2 + 1:]:
                        layers_dec.append(nn.Linear(prev, d))
                        layers_dec.append(nn.ReLU())
                        prev = d
                    layers_dec.append(nn.Linear(prev, input_dim))
                    self.decoder = nn.Sequential(*layers_dec)

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            self.model = AE(self.hidden_dims)
            self._backend = "torch"
        except ImportError:
            logger.info("PyTorch not available — using numpy autoencoder fallback")
            self._backend = "numpy"

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self._build_model(X_scaled.shape[1])

        if self._backend == "torch":
            self._fit_torch(X_scaled)
        else:
            self._fit_numpy(X_scaled)

        # Set threshold as 95th percentile of training reconstruction error
        errors = self._reconstruction_error(X_scaled)
        self.threshold = np.percentile(errors, 95)
        self.fitted = True
        return self

    def _fit_torch(self, X: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        tensor = torch.FloatTensor(X)
        loader = DataLoader(TensorDataset(tensor, tensor),
                            batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def _fit_numpy(self, X: np.ndarray):
        """Simple PCA-based reconstruction as fallback."""
        from sklearn.decomposition import PCA
        bottleneck = min(self.hidden_dims)
        self._pca = PCA(n_components=min(bottleneck, X.shape[1], X.shape[0]))
        self._pca.fit(X)

    def _reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        if self._backend == "torch":
            import torch
            self.model.eval()
            with torch.no_grad():
                tensor = torch.FloatTensor(X_scaled)
                output = self.model(tensor).numpy()
            return np.mean((X_scaled - output) ** 2, axis=1)
        else:
            transformed = self._pca.transform(X_scaled)
            reconstructed = self._pca.inverse_transform(transformed)
            return np.mean((X_scaled - reconstructed) ** 2, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores normalized to [0, 1]."""
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        errors = self._reconstruction_error(X_scaled)
        # Normalize using training threshold
        scores = errors / (self.threshold + 1e-10)
        return np.clip(scores, 0, 1)


# Ensemble Anomaly Scorer

class EnsembleAnomalyScorer:


    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or load_config()
        self.if_detector = IsolationForestDetector(self.cfg)
        self.xgb_detector = XGBoostDetector(self.cfg)
        self.ae_detector = AutoencoderDetector(self.cfg)
        self.lof_detector = LOFDetector(self.cfg)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        logger.info("Fitting ensemble anomaly detectors...")
        self.if_detector.fit(X)
        self.ae_detector.fit(X)
        self.lof_detector.fit(X)
        if y is not None and len(np.unique(y)) > 1:
            self.xgb_detector.fit(X, y)
        else:
            logger.warning("No labels or single class — XGBoost not fitted")
        return self

    def score(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        scores = {}
        scores["isolation_forest"] = self.if_detector.score(X)
        scores["autoencoder"] = self.ae_detector.score(X)
        scores["lof"] = self.lof_detector.score(X)

        if self.xgb_detector.fitted:
            scores["xgboost"] = self.xgb_detector.score(X)
        else:
            scores["xgboost"] = scores["isolation_forest"]  # fallback

        # Weighted ensemble (IF + XGB + AE)
        w_if = self.if_detector.weight     # 0.35
        w_xgb = self.xgb_detector.weight   # 0.45
        w_ae = self.ae_detector.weight      # 0.20

        scores["ensemble"] = (
            w_if * scores["isolation_forest"]
            + w_xgb * scores["xgboost"]
            + w_ae * scores["autoencoder"]
        )

        return scores

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all models against ground truth."""
        all_scores = self.score(X)
        results = {}

        for model_name, scores in all_scores.items():
            y_pred = (scores >= threshold).astype(int)
            try:
                auroc = roc_auc_score(y_true, scores)
            except ValueError:
                auroc = 0.0

            results[model_name] = {
                "auroc": auroc,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "avg_precision": average_precision_score(y_true, scores)
                if len(np.unique(y_true)) > 1 else 0.0,
            }

        return results
