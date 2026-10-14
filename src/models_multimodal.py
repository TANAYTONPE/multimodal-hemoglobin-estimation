"""
models_multimodal.py
--------------------
Architectures and regression heads for Single-Modal, Dual-Modal,
and Tri-Modal Hemoglobin Estimation.

Includes:
- Baseline Color Regressor (RGB, HSV, LAB linear model)
- Ridge & MLP Regression Heads for single-modal embeddings
- Model 4: Fingernail + Skin Feature Fusion Head
- Model 5: Fingernail + Conjunctiva Feature Fusion Head
- Model 6: Full Three-Branch Multimodal Architecture (Nail + Skin + Conjunctiva)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── PyTorch Regression Heads ───────────────────────────────────────────────────

class MLPRegressionHead(nn.Module):
    """
    Lightweight, heavily regularized neural regression head with residual connection.
    Predicts scalar hemoglobin in g/dL from feature embeddings.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)  # (B,)


class MultimodalFusionModel(nn.Module):
    """
    Model 4 / Model 5 Dual-Branch Feature Fusion Architecture.
    Fuses two modal representations (e.g. Nail + Skin) via concatenated projection.
    """
    def __init__(self, in_dim1: int = 1296, in_dim2: int = 1296, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        fused_dim = in_dim1 + in_dim2
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([feat1, feat2], dim=-1)
        out = self.fusion(fused)
        return out.squeeze(-1)  # (B,)


class TriModalHemoglobinNet(nn.Module):
    """
    Model 6: MAIN PROPOSED THREE-BRANCH MULTIMODAL ARCHITECTURE.

    Architecture:
    Branch 1: Fingernail Image -> CNN Encoder -> Nail Embedding (d=1296)
    Branch 2: Skin Image       -> CNN Encoder -> Skin Embedding (d=1296)
    Branch 3: Conjunctiva Image-> CNN Encoder -> Conjunctiva Embedding (d=1296)
    FUSE ALL THREE:
    [Nail || Skin || Conjunctiva] (d = 3888)
        -> Learned Fusion Linear Layer (3888 -> 256)
        -> LayerNorm + GELU + Dropout(0.25)
        -> Dense(256 -> 64) + GELU
        -> Regressor(64 -> 1)
        -> Predicted Hemoglobin in g/dL
    """
    def __init__(
        self,
        embed_dim: int = 1296,
        fusion_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()
        self.embed_dim = embed_dim
        total_dim = embed_dim * 3

        self.fusion_projection = nn.Sequential(
            nn.BatchNorm1d(total_dim),
            nn.Dropout(p=dropout),
            nn.Linear(total_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        nail_feat: torch.Tensor,
        skin_feat: torch.Tensor,
        conj_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with three modality feature tensors.
        Each tensor has shape (B, embed_dim).
        Returns shape (B, 1) continuous hemoglobin values in g/dL.
        """
        fused = torch.cat([nail_feat, skin_feat, conj_feat], dim=-1)
        pred_hb = self.fusion_projection(fused)
        return pred_hb


# ── Sklearn / Metric Helpers ───────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: MAE, RMSE, R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "n_samples": len(y_true),
    }


class SklearnHbRegressor:
    """
    Standardized, regularized Ridge regression estimator with feature scaling.
    Guarantees stable, non-exploding continuous hemoglobin predictions.
    """
    def __init__(self, alpha: float = 100.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        # Clinical plausibility bounds: 3.0 to 22.0 g/dL
        return np.clip(preds, 3.0, 22.0)
