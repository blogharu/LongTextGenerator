import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self): ...

    def forward(self, x): ...


class RMSNorm(nn.Module):
    def __init__(self): ...

    def forward(self, x): ...


class SelfAttention(nn.Module):
    def __init__(self): ...

    def forward(self, x): ...


class MoEGenreGate(nn.Module):
    def __init__(self): ...

    def forward(self, x, genre_emb): ...


class MixtralGenreGateModel(nn.Module):
    def __init__(
        self,
        num_layers,  # num
    ): ...

    def forward(self, x, genre_emb): ...
