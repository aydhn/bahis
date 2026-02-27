"""
multi_task_backbone.py – Multi-Task Learning (MTL) Architecture.

A unified Deep Learning brain that predicts multiple market outcomes
(1X2, Goals, Corners) simultaneously, leveraging shared feature representations.
Supports dynamic loss weighting (Homoscedastic Uncertainty) to balance task importance.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found – MTL will run in heuristic mode.")


if TORCH_AVAILABLE:
    class MTLNet(nn.Module):
        """
        Shared Backbone + Task-Specific Heads.
        Includes learnable uncertainty weights for loss balancing.
        """
        def __init__(self, input_dim: int = 12, hidden: int = 128):
            super().__init__()
            # Shared Encoder (Backbone)
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(0.3),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(0.2),
            )

            # 1. Outcome Head (Classification: Home/Draw/Away)
            self.head_outcome = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Softmax(dim=-1),
            )

            # 2. Goals Head (Regression: Expected Goals)
            self.head_goals = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus(),  # Goals must be positive
            )

            # 3. Corners Head (Regression: Expected Corners)
            self.head_corners = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus(),  # Corners must be positive
            )

            # Learnable Uncertainty Parameters (Log Variance) for Loss Balancing
            # sigma_outcome, sigma_goals, sigma_corners
            self.log_vars = nn.Parameter(torch.zeros(3))

        def forward(self, x):
            shared_features = self.backbone(x)
            return {
                "outcome_probs": self.head_outcome(shared_features),
                "expected_goals": self.head_goals(shared_features),
                "expected_corners": self.head_corners(shared_features),
            }

        def uncertainty_weighted_loss(self, preds, targets):
            """
            Multi-task loss with homoscedastic uncertainty weighting.
            Loss = (1/2σ₁²)*L₁ + log(σ₁) + (1/2σ₂²)*L₂ + log(σ₂) + ...
            """
            # Outcome Loss (Cross Entropy)
            loss_outcome = nn.CrossEntropyLoss()(preds["outcome_probs"], targets["outcome"])
            precision_outcome = torch.exp(-self.log_vars[0])
            weighted_outcome = precision_outcome * loss_outcome + 0.5 * self.log_vars[0]

            # Goals Loss (MSE)
            loss_goals = nn.MSELoss()(preds["expected_goals"], targets["goals"])
            precision_goals = torch.exp(-self.log_vars[1])
            weighted_goals = precision_goals * loss_goals + 0.5 * self.log_vars[1]

            # Corners Loss (MSE)
            loss_corners = nn.MSELoss()(preds["expected_corners"], targets["corners"])
            precision_corners = torch.exp(-self.log_vars[2])
            weighted_corners = precision_corners * loss_corners + 0.5 * self.log_vars[2]

            return weighted_outcome + weighted_goals + weighted_corners


class MultiTaskBackbone:
    """
    Manager class for the MTL Network. Handles training, inference, and persistence.
    """

    FEATURE_KEYS = [
        "home_odds", "draw_odds", "away_odds",
        "over25_odds", "under25_odds",
        "home_xg", "away_xg", "home_xga", "away_xga",
        "home_win_rate", "away_win_rate", "odds_volatility",
        "home_possession", "away_possession", "home_shots", "away_shots"
    ]

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._device = "cpu"
        self.input_dim = len(self.FEATURE_KEYS)

        if TORCH_AVAILABLE:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = MTLNet(input_dim=self.input_dim)
            self._model.to(self._device)

            if model_path:
                try:
                    self._model.load_state_dict(torch.load(model_path, map_location=self._device))
                    self._model.eval()
                    logger.info(f"MTL model loaded from: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load MTL model: {e}")

        logger.debug(f"MultiTaskBackbone initialized on {self._device}.")

    def _extract_features_batch(self, features: pl.DataFrame) -> np.ndarray:
        """Extracts and normalizes features from a DataFrame batch."""
        # Ensure all columns exist, fill missing with 0
        df_subset = features.select([
            pl.col(k).fill_null(0.0) if k in features.columns else pl.lit(0.0).alias(k)
            for k in self.FEATURE_KEYS
        ])

        # Convert to numpy array (N_samples, N_features)
        return df_subset.to_numpy().astype(np.float32)

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        Runs inference for a batch of matches.
        Returns DataFrame with predictions for Outcome, Goals, and Corners.
        """
        if features.is_empty():
            return pl.DataFrame()

        # Vectorized feature extraction
        X_batch = self._extract_features_batch(features)
        match_ids = features["match_id"].to_list() if "match_id" in features.columns else [""] * len(features)

        if TORCH_AVAILABLE and self._model is not None:
            preds = self._predict_torch_batch(X_batch)
        else:
            preds = self._predict_heuristic_batch(features)

        # Combine into result DataFrame
        results_df = pl.DataFrame({
            "match_id": match_ids,
            "mtl_prob_home": preds["prob_home"],
            "mtl_prob_draw": preds["prob_draw"],
            "mtl_prob_away": preds["prob_away"],
            "mtl_expected_goals": preds["expected_goals"],
            "mtl_expected_corners": preds["expected_corners"],
            "mtl_confidence": preds["confidence"],
        })

        return results_df

    def predict_batch_numpy(self, X_batch: np.ndarray, match_ids: list[str] | None = None) -> pl.DataFrame:
        """
        Zero-Copy friendly inference.
        Accepts numpy array directly.
        """
        if X_batch.size == 0:
            return pl.DataFrame()

        # If SHM passed more rows than we need, we trust match_ids length if available, else slice
        if match_ids and len(match_ids) < X_batch.shape[0]:
             X_batch = X_batch[:len(match_ids)]
        elif match_ids is None:
            # Generate placeholder IDs if not provided
            match_ids = ["" for _ in range(X_batch.shape[0])]

        # Ensure input dimensions match expectations (truncate if SHM is padded)
        # We assume X_batch columns are at least input_dim
        if X_batch.shape[1] > self.input_dim:
            X_batch = X_batch[:, :self.input_dim]

        if TORCH_AVAILABLE and self._model is not None:
            preds = self._predict_torch_batch(X_batch)
        else:
            preds = self._predict_heuristic_batch_numpy(X_batch)

        results_df = pl.DataFrame({
            "match_id": match_ids,
            "mtl_prob_home": preds["prob_home"],
            "mtl_prob_draw": preds["prob_draw"],
            "mtl_prob_away": preds["prob_away"],
            "mtl_expected_goals": preds["expected_goals"],
            "mtl_expected_corners": preds["expected_corners"],
            "mtl_confidence": preds["confidence"],
        })

        return results_df

    def _predict_torch_batch(self, X_batch: np.ndarray) -> dict:
        """PyTorch batched inference."""
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(X_batch, dtype=torch.float32).to(self._device)
            out = self._model(x)

            probs = out["outcome_probs"].cpu().numpy() # (N, 3)
            goals = out["expected_goals"].cpu().numpy().flatten() # (N,)
            corners = out["expected_corners"].cpu().numpy().flatten() # (N,)

        # Calculate confidence (Max probability)
        # Using max probability as confidence metric: higher max means more confident model
        confidence = np.max(probs, axis=1)

        return {
            "prob_home": probs[:, 0],
            "prob_draw": probs[:, 1],
            "prob_away": probs[:, 2],
            "expected_goals": goals,
            "expected_corners": corners,
            "confidence": confidence,
        }

    def _predict_heuristic_batch(self, features: pl.DataFrame) -> dict:
        """Fallback logic when PyTorch is not available (Vectorized from DataFrame)."""
        n = features.height

        # Simple Odds Implied Probability
        h = features.get_column("home_odds").fill_null(2.5).to_numpy()
        d = features.get_column("draw_odds").fill_null(3.2).to_numpy()
        a = features.get_column("away_odds").fill_null(2.8).to_numpy()

        # Avoid division by zero
        h = np.maximum(h, 1.01)
        d = np.maximum(d, 1.01)
        a = np.maximum(a, 1.01)

        probs_h = 1.0 / h
        probs_d = 1.0 / d
        probs_a = 1.0 / a

        total_probs = probs_h + probs_d + probs_a
        probs_h /= total_probs
        probs_d /= total_probs
        probs_a /= total_probs

        # Simple xG Sum
        xg_h = features.get_column("home_xg").fill_null(1.35).to_numpy() if "home_xg" in features.columns else np.full(n, 1.35)
        xg_a = features.get_column("away_xg").fill_null(1.15).to_numpy() if "away_xg" in features.columns else np.full(n, 1.15)

        xg = xg_h + xg_a

        # Corners correlation with xG (~4.5 corners per goal is a rough heuristic)
        corners = xg * 4.5

        # Confidence is max prob
        confidence = np.maximum(probs_h, np.maximum(probs_d, probs_a))

        return {
            "prob_home": probs_h,
            "prob_draw": probs_d,
            "prob_away": probs_a,
            "expected_goals": xg,
            "expected_corners": corners,
            "confidence": confidence,
        }

    def _predict_heuristic_batch_numpy(self, X_batch: np.ndarray) -> dict:
        """Fallback logic when PyTorch is not available (Vectorized from Numpy)."""
        # Feature Mapping (Must match FEATURE_KEYS order)
        # 0: home_odds, 1: draw_odds, 2: away_odds
        # 5: home_xg, 6: away_xg

        h = X_batch[:, 0]
        d = X_batch[:, 1]
        a = X_batch[:, 2]

        h = np.maximum(h, 1.01)
        d = np.maximum(d, 1.01)
        a = np.maximum(a, 1.01)

        probs_h = 1.0 / h
        probs_d = 1.0 / d
        probs_a = 1.0 / a

        total_probs = probs_h + probs_d + probs_a
        # Avoid zero total (if all odds are huge? unlikely with max 1.01)
        probs_h /= total_probs
        probs_d /= total_probs
        probs_a /= total_probs

        # xG
        xg_h = X_batch[:, 5]
        xg_a = X_batch[:, 6]

        # Use defaults if 0
        xg_h = np.where(xg_h <= 0, 1.35, xg_h)
        xg_a = np.where(xg_a <= 0, 1.15, xg_a)

        xg = xg_h + xg_a
        corners = xg * 4.5

        confidence = np.maximum(probs_h, np.maximum(probs_d, probs_a))

        return {
            "prob_home": probs_h,
            "prob_draw": probs_d,
            "prob_away": probs_a,
            "expected_goals": xg,
            "expected_corners": corners,
            "confidence": confidence,
        }

    def train(self, X: np.ndarray, targets: dict[str, np.ndarray], epochs: int = 50, lr: float = 1e-3):
        """
        Train the MTL model.
        targets dict should contain: 'outcome' (long), 'goals' (float), 'corners' (float)
        """
        if not TORCH_AVAILABLE or self._model is None:
            logger.warning("PyTorch not available, skipping training.")
            return

        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        # Convert data to tensors
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_outcome = torch.tensor(targets["outcome"], dtype=torch.long).to(self._device)
        y_goals = torch.tensor(targets["goals"], dtype=torch.float32).unsqueeze(1).to(self._device)
        y_corners = torch.tensor(targets["corners"], dtype=torch.float32).unsqueeze(1).to(self._device)

        target_dict = {
            "outcome": y_outcome,
            "goals": y_goals,
            "corners": y_corners
        }

        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self._model(X_t)

            # Use the dynamic uncertainty weighted loss
            loss = self._model.uncertainty_weighted_loss(preds, target_dict)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"MTL Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                            f"Sigmas: {torch.exp(self._model.log_vars).detach().cpu().numpy()}")

        self._model.eval()
