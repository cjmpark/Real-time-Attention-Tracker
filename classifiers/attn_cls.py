import torch.nn as nn


class AttentionNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Linear(32,1)
        )

    def forward(self, x):
        return self.model(x)

