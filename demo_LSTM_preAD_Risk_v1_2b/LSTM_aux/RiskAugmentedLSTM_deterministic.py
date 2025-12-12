# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 19:49:31 2025

@author: cplatero
"""

import torch
import torch.nn as nn
import numpy as np


import random
import os



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)





# -----------------------------
# 1. Capa lineal monotónica
# -----------------------------


class MonotonicLinear(nn.Module):
    def __init__(self, in_features, out_features, monotonic_indices):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.monotonic_indices = monotonic_indices

    def forward(self, x):
        weight = self.weight_raw.clone()
        weight[:, self.monotonic_indices] = torch.abs(weight[:, self.monotonic_indices])
        return torch.matmul(x, weight.t()) + self.bias


class NeuroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3, num_layers=2):
        super(NeuroLSTM, self).__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, input_size)  # sólo predice medidas

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x_last = self.dropout(lstm_out[:, -1])
        residual = self.fc(x_last)
        # return x[:, -1, :] + residual  # devuelve medidas futuras
        x_last_input = x[:, -1, :].clone()
        return x_last_input + residual




class DxClassifier(nn.Module):
    def __init__(self, input_size):
        super(DxClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_predicted):
        return self.net(x_predicted)  # predicción diagnóstico



# -----------------------------
# 2. Red integrada sin imputación
# -----------------------------
class RiskAugmentedJointLSTMClassifier(nn.Module):
    def __init__(self, input_size, static_indices, hidden_size, dropout_rate=0.3, num_layers=2):
        super().__init__()
        self.static_indices = static_indices
        self.dynamic_indices = [i for i in range(input_size) if i not in static_indices]
        self.input_dynamic_size = len(self.dynamic_indices)

        # Riesgo basal
        self.f_age = nn.Sequential(MonotonicLinear(1, 1, [0]), nn.Softplus())
        self.f_apoe = nn.Sequential(MonotonicLinear(1, 1, [0]), nn.Softplus())
        self.f_edu = nn.Sequential(MonotonicLinear(1, 1, [0]), nn.Softplus())
        self.f_sex = nn.Linear(1, 1)
        self.output_risk = nn.Sigmoid()

        self.lstm_model = NeuroLSTM(input_size=self.input_dynamic_size,
                                    hidden_size=hidden_size,
                                    dropout_rate=dropout_rate,
                                    num_layers=num_layers)
        self.classifier = DxClassifier(input_size=self.input_dynamic_size+1) # risk
        # self.classifier = DxClassifier(input_size=self.input_dynamic_size) # sin risk

    def forward(self, x, diag_init):
        B, T, D = x.shape
        device = x.device
        x_static = x[:, 0, self.static_indices]  # [B, 4]
        x_dynamic = x[:, :, self.dynamic_indices]  # [B, T, D_dyn]

        # Riesgo basal
        age = x_static[:, [0]]
        edu = -x_static[:, [1]]
        apoe = x_static[:, [2]]
        sex = x_static[:, [3]]

        diag_init = diag_init.view(-1, 1).to(device)
        mask_cn = (diag_init == 0).float()

        age = age * mask_cn
        edu = edu * mask_cn
        apoe = apoe * mask_cn
        sex = sex * mask_cn

        risk = self.f_age(age) + self.f_apoe(apoe) + self.f_edu(edu) + self.f_sex(sex)
        risk = self.output_risk(risk) * mask_cn  # [B, 1]

        pred_measures = self.lstm_model(x_dynamic)  # [B, D_dyn]
        
        # Concatenar riesgo basal a la predicción de medidas para el diagnóstico
        x_diagnosis = torch.cat([pred_measures, risk], dim=-1)  # [B, D_dyn + 1] risk
        # x_diagnosis = pred_measures # sin risk
        
        pred_dx = self.classifier(x_diagnosis)
        return pred_measures, pred_dx




def RiskAugmentedLSTM_deterministic(input_size, static_indices, hidden_size, dropout_rate=0.3, num_layers=2, seed=42):
    set_seed(seed)
    model = RiskAugmentedJointLSTMClassifier(
        input_size=input_size,
        static_indices=static_indices,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_layers=num_layers
    )
    initialize_weights(model)
    return model

