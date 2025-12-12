# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:59:38 2025

@author: cplatero
"""

import torch
import torch.nn as nn

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim))  # [input_dim]
        self.b = nn.Parameter(torch.randn(input_dim))  # [input_dim]

    def forward(self, x0, x):
        # x0, x: [batch_size, input_dim]
        weighted = torch.matmul(x, self.w).unsqueeze(1)  # [batch_size, 1]
        return x0 * weighted + self.b + x  # broadcasting sobre input_dim

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        x0 = x
        for layer in self.cross_layers:
            x = layer(x0, x)
        return x

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], input_dim)
        )

    def forward(self, x):
        return self.layers(x)

class EDCFModule(nn.Module):
    def __init__(self, input_dim, embedding_dims={}):
        super().__init__()
        self.input_dim = input_dim
        self.deep = DeepNetwork(input_dim)
        self.cross = CrossNetwork(input_dim)
        self.cross_deep = nn.Sequential(self.cross, self.deep)
        self.deep_cross = nn.Sequential(self.deep, self.cross)
        self.output = nn.Linear(input_dim, input_dim)  # final mapping

    def forward(self, x_dense, x_cat):
        # x_dense: [B, input_dim]
        # x_cat is ignored in this binary version
        x = x_dense
        out_deep = self.deep(x)
        out_cross = self.cross(x)
        out_cd = self.cross_deep(x)
        out_dc = self.deep_cross(x)
        fused = out_deep + out_cross + out_cd + out_dc  # fusion
        fused = torch.stack([out_deep, out_cross, out_cd, out_dc], dim=0).sum(dim=0)
        return self.output(fused)  # output shape: [B, input_dim]

