"""Relation Classifier head for the edge-featured-GAT architecture.

Takes a single entity-pair feature vector (the output of the upstream
2-layer edge-featured GAT, not implemented here) and maps it to per-relation
scores, matching the team's architecture diagram:

    768-d Feature -> Linear(768, 768) -> GELU -> Dropout(0.1) -> Linear(768, 97)
    -> Relation Logits (96 relations + NA) -> Sigmoid

Class index 0 is NA/no-relation (same convention as `Scripts.atlop.losses.ATLoss`
and `data.docred_io.build_rel2id`, where the 96 real P-codes are 1..96). Sigmoid
(not softmax) because a pair can hold more than one relation at once.

Loss is out of scope here (BCEWithLogitsLoss + evidence contrastive loss are a
separate piece of work) -- use `logits`, not `probs`, if training with
`nn.BCEWithLogitsLoss`, since it applies sigmoid internally and double-applying
it would break gradients.
"""

import torch
import torch.nn as nn


class RelationClassifier(nn.Module):
    def __init__(self, hidden_size: int = 768, num_labels: int = 97, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, pair_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pair_features: (n_pairs, hidden_size). Returns (logits, probs), both
        (n_pairs, num_labels)."""
        logits = self.mlp(pair_features)
        probs = torch.sigmoid(logits)
        return logits, probs
