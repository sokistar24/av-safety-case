"""
cte_model.py
Small TaxiNet-style CNN for CTE regression.

Follows the paper's TaxiNet: 5 convolutional layers + 3 dense layers (100/50/10)
with ELU activations, single continuous output (cte). Dropout is added before the
dense layers so we can do MC-Dropout uncertainty estimation later (the Level-1
run-time handover): keep dropout ACTIVE at inference, run N forward passes, and use
the spread of cte predictions as the model's uncertainty.

Kept deliberately small and transparent (vs. a big pretrained backbone) because:
  - it matches the paper for a faithful replication,
  - it trains fast on a 4 GB laptop GPU,
  - its named dense layers are easy to introspect for the optional Level-2
    activation-rule guard.

Input: (B, 3, 80, 160)  ->  Output: (B, 1) continuous cte.
"""

import torch
import torch.nn as nn


class TaxiNetCTE(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        # 5 conv layers (NVIDIA/TaxiNet-style), ELU activations, no maxpool.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),   # ->  (24, 38, 78)
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),  # ->  (36, 17, 37)
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),  # ->  (48, 7, 17)
            nn.Conv2d(48, 64, kernel_size=3), nn.ELU(),            # ->  (64, 5, 15)
            nn.Conv2d(64, 64, kernel_size=3), nn.ELU(),            # ->  (64, 3, 13)
        )
        # dense head 100 -> 50 -> 10 -> 1, with dropout for MC-Dropout uncertainty
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.LazyLinear(100), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),          # single continuous cte output
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.drop(x)
        return self.fc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def mc_dropout_predict(model, x, n_samples=30):
    """
    MC-Dropout inference for uncertainty (used by the Level-1 handover later).
    Keeps dropout ACTIVE (model.train() on the dropout layers) and runs n forward
    passes. Returns (mean_cte, std_cte) per input; std is the uncertainty signal.
    """
    was_training = model.training
    model.train()  # activate dropout
    preds = torch.stack([model(x) for _ in range(n_samples)], dim=0)  # (n, B, 1)
    if not was_training:
        model.eval()
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    return mean, std


if __name__ == "__main__":
    # self-test: build, param count, forward pass, MC-dropout spread on a dummy batch
    model = TaxiNetCTE(dropout=0.3)
    x = torch.randn(4, 3, 80, 160)
    # initialize LazyLinear by running one forward pass
    out = model(x)
    print(f"Parameters: {count_parameters(model)/1e6:.2f} M")
    print(f"Output shape: {tuple(out.shape)}  (expect (4, 1))")
    mean, std = mc_dropout_predict(model, x, n_samples=10)
    print(f"MC-dropout mean shape: {tuple(mean.shape)}, std shape: {tuple(std.shape)}")
    print(f"Example uncertainty (std) values: {std.squeeze().tolist()}")
    print("cte_model.py self-test passed.")
