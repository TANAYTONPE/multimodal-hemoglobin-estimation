"""
models.py
---------
Modular architectures for Single-Modal and Multimodal Hemoglobin Estimation:
- Model 1: Fingernail Hb Estimator (EfficientNet-B0 backbone + Regression Head)
- Model 2: Skin Hb Estimator (EfficientNet-B0 backbone + Regression Head)
- Model 3: Conjunctival Hb Estimator (Modular branch interface, training pending)
- Model 4: Multimodal Fingernail + Skin Fusion (Concatenated embeddings + Fusion Head)
- Model 5: Multimodal Fingernail + Conjunctiva Fusion (Modular interface)
- Model 6: Main Proposed Multimodal Model (Three-branch: Fingernail + Skin + Conjunctiva Fusion)
"""

import torch
import torch.nn as nn
from torchvision import models

# ── Modality CNN Encoder ───────────────────────────────────────────────────────
class ModalityEncoder(nn.Module):
    """
    CNN Feature Encoder using frozen EfficientNet-B0.
    Produces 1280-dimensional global embedding vector.
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        
        if freeze:
            for p in backbone.parameters():
                p.requires_grad = False
        
        self.backbone = backbone
        self.out_dim = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Regression Head ────────────────────────────────────────────────────────────
class RegressionHead(nn.Module):
    """
    Multilayer perceptron regression head mapping feature embeddings to Hb (g/dL).
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Model 1: Fingernail Model ──────────────────────────────────────────────────
class FingernailModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.encoder = ModalityEncoder(freeze=freeze_backbone)
        self.head = RegressionHead(in_features=self.encoder.out_dim)

    def forward(self, nail_img: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(nail_img)
        return self.head(feat)


# ── Model 2: Skin Model ────────────────────────────────────────────────────────
class SkinModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.encoder = ModalityEncoder(freeze=freeze_backbone)
        self.head = RegressionHead(in_features=self.encoder.out_dim)

    def forward(self, skin_img: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(skin_img)
        return self.head(feat)


# ── Model 3: Conjunctival Model (Modular Branch) ───────────────────────────────
class ConjunctivalModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.encoder = ModalityEncoder(freeze=freeze_backbone)
        self.head = RegressionHead(in_features=self.encoder.out_dim)

    def forward(self, eye_img: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(eye_img)
        return self.head(feat)


# ── Model 4: Multimodal Fingernail + Skin Fusion ──────────────────────────────
class NailSkinFusionModel(nn.Module):
    """
    Dual-modality fusion network:
    Fingernail (1280) + Skin (1280) -> Concatenation (2560) -> Fusion Regression Head -> Hb
    """
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.nail_encoder = ModalityEncoder(freeze=freeze_backbone)
        self.skin_encoder = ModalityEncoder(freeze=freeze_backbone)
        fusion_dim = self.nail_encoder.out_dim + self.skin_encoder.out_dim  # 2560
        self.fusion_head = RegressionHead(in_features=fusion_dim, hidden_dim=256)

    def forward(self, nail_img: torch.Tensor, skin_img: torch.Tensor) -> torch.Tensor:
        nail_feat = self.nail_encoder(nail_img)
        skin_feat = self.skin_encoder(skin_img)
        fused = torch.cat([nail_feat, skin_feat], dim=-1)
        return self.fusion_head(fused)

    def forward_features(self, nail_feat: torch.Tensor, skin_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass directly from pre-extracted feature tensors."""
        fused = torch.cat([nail_feat, skin_feat], dim=-1)
        return self.fusion_head(fused)


# ── Model 5: Multimodal Fingernail + Conjunctiva Fusion ────────────────────────
class NailConjunctivaFusionModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.nail_encoder = ModalityEncoder(freeze=freeze_backbone)
        self.eye_encoder = ModalityEncoder(freeze=freeze_backbone)
        fusion_dim = self.nail_encoder.out_dim + self.eye_encoder.out_dim  # 2560
        self.fusion_head = RegressionHead(in_features=fusion_dim, hidden_dim=256)

    def forward(self, nail_img: torch.Tensor, eye_img: torch.Tensor) -> torch.Tensor:
        nail_feat = self.nail_encoder(nail_img)
        eye_feat = self.eye_encoder(eye_img)
        fused = torch.cat([nail_feat, eye_feat], dim=-1)
        return self.fusion_head(fused)


# ── Model 6: MAIN PROPOSED MODEL (Three-Branch Multimodal Fusion) ──────────────
class MainProposedMultimodalModel(nn.Module):
    """
    Full Multimodal Non-Invasive Hemoglobin Estimator:
    - Branch 1: Fingernail Image -> EfficientNet-B0 (1280)
    - Branch 2: Skin Image       -> EfficientNet-B0 (1280)
    - Branch 3: Conjunctiva Image -> EfficientNet-B0 (1280)
    Fusion Head: Concatenated embeddings (3840) -> Dense MLP -> Continuous Hb (g/dL)
    """
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.nail_branch = ModalityEncoder(freeze=freeze_backbone)
        self.skin_branch = ModalityEncoder(freeze=freeze_backbone)
        self.eye_branch  = ModalityEncoder(freeze=freeze_backbone)
        
        total_fusion_dim = 1280 * 3  # 3840
        self.fusion_head = nn.Sequential(
            nn.Linear(total_fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, nail_img: torch.Tensor, skin_img: torch.Tensor, eye_img: torch.Tensor) -> torch.Tensor:
        nail_emb = self.nail_branch(nail_img)
        skin_emb = self.skin_branch(skin_img)
        eye_emb  = self.eye_branch(eye_img)
        fused = torch.cat([nail_emb, skin_emb, eye_emb], dim=-1)
        return self.fusion_head(fused).squeeze(-1)

    def forward_features(self, nail_feat: torch.Tensor, skin_feat: torch.Tensor, eye_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([nail_feat, skin_feat, eye_feat], dim=-1)
        return self.fusion_head(fused).squeeze(-1)


# ── Smoke Test ─────────────────────────────────────────────────────────────────
def run_model6_smoke_test():
    """Verifies architectural validity and tensor shape propagation for Model 6."""
    print("\n--- Running Model 6 Synthetic Tensor-Shape Smoke Test ---")
    model = MainProposedMultimodalModel(freeze_backbone=True)
    batch_size = 2
    dummy_nail = torch.randn(batch_size, 3, 224, 224)
    dummy_skin = torch.randn(batch_size, 3, 224, 224)
    dummy_eye  = torch.randn(batch_size, 3, 224, 224)
    
    out = model(dummy_nail, dummy_skin, dummy_eye)
    print(f"Input batch size: {batch_size} (Nail: {dummy_nail.shape}, Skin: {dummy_skin.shape}, Eye: {dummy_eye.shape})")
    print(f"Output shape: {out.shape}")
    assert out.shape == torch.Size([batch_size]), f"Expected shape {[batch_size]}, got {out.shape}"
    print("Smoke Test Passed: Model 6 three-modality tensor forward propagation is valid!\n")
    return True


if __name__ == "__main__":
    run_model6_smoke_test()
