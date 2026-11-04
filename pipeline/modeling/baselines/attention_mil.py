"""
CLAM-SB (Clustering-constrained Attention MIL, Single Branch)
Reference: Lu et al., Nature Biomedical Engineering, 2021

Input:  (N, feature_dim) tile embeddings  — N varies per slide
Output: binary classification logit (ER+/-)

Architecture:
  1. Feature encoder: Linear(feature_dim→512) → ReLU → Dropout
  2. Gated attention: U (tanh) × V (sigmoid) → score → softmax weights
  3. Aggregation: weighted sum → (1, 512) slide representation
  4. Classifier: Linear(512→1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, dim: int = 512, att_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.U = nn.Sequential(nn.Linear(dim, att_dim), nn.Tanh(), nn.Dropout(dropout))
        self.V = nn.Sequential(nn.Linear(dim, att_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(att_dim, 1)

    def forward(self, h: torch.Tensor):
        """h: (N, dim) → weights: (N, 1), attended: (1, dim)"""
        scores = self.w(self.U(h) * self.V(h))          # (N, 1)
        weights = F.softmax(scores, dim=0)               # (N, 1)
        attended = (weights * h).sum(dim=0, keepdim=True)  # (1, dim)
        return attended, weights


class CLAMSB(nn.Module):
    """CLAM Single Branch — binary phenotype prediction from WSI tile embeddings."""

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        att_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(hidden_dim, att_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (N, feature_dim) tile embeddings
        returns: logit (scalar), attention_weights (N, 1)
        """
        h = self.encoder(x)                       # (N, hidden_dim)
        attended, weights = self.attention(h)      # (1, hidden_dim), (N, 1)
        logit = self.classifier(attended).squeeze(-1)  # (1,)
        return logit, weights


class CLAMMB(nn.Module):
    """CLAM Multi-Branch — multi-class phenotype prediction (e.g. PAM50, 5 classes).

    Reference: Lu et al., Nature Biomedical Engineering, 2021 (CLAM-MB variant).
    Unlike CLAM-SB, each class has its own gated-attention branch and its own
    classifier head, so the model learns class-specific tile importance
    (e.g. which tiles matter for "LumA" may differ from "Basal").
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        att_dim: int = 256,
        dropout: float = 0.25,
        num_classes: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_branches = nn.ModuleList(
            [GatedAttention(hidden_dim, att_dim, dropout) for _ in range(num_classes)]
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_classes)]
        )

    def forward(self, x: torch.Tensor):
        """
        x: (N, feature_dim) tile embeddings
        returns: logits (num_classes,), attention_weights (num_classes, N)
        """
        h = self.encoder(x)  # (N, hidden_dim)
        logits, all_weights = [], []
        for k in range(self.num_classes):
            attended, weights = self.attention_branches[k](h)  # (1, hidden_dim), (N, 1)
            logit_k = self.classifiers[k](attended).squeeze()   # scalar
            logits.append(logit_k)
            all_weights.append(weights.squeeze(-1))             # (N,)
        logits = torch.stack(logits)          # (num_classes,)
        all_weights = torch.stack(all_weights)  # (num_classes, N)
        return logits, all_weights
