# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 13:22:35 2025

@author: cplatero
"""


# --- Módulos GAN para Imputación (con la corrección) ---
import torch
import torch.nn as nn


# --- Importaciones de tus módulos auxiliares ---
from LSTM_aux.RiskAugmentedLSTM_deterministic import MonotonicLinear, DxClassifier, set_seed, initialize_weights, NeuroLSTM 




# --- Módulos GAN para Imputación (con la corrección) ---

class Generator(nn.Module):
    # ... (Sin cambios)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    def forward(self, x, m):
        inp = torch.cat([x, m], dim=-1)
        return self.net(inp)

class Discriminator(nn.Module):
    # ... (Sin cambios)
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x, h):
        inp = torch.cat([x, h], dim=-1)
        return self.net(inp)

class TSGAIN_Module(nn.Module):
    """Módulo TSGAIN con la corrección del tipo de tensor."""
    def __init__(self, dynamic_dim, hidden_dim=64):
        super(TSGAIN_Module, self).__init__()
        self.generator = Generator(dynamic_dim, hidden_dim, dynamic_dim)
        self.discriminator = Discriminator(dynamic_dim, hidden_dim)

    def forward(self, x, mask):
        # x_placeholder = torch.nan_to_num(x, nan=0.0)
        # generated_values = self.generator(x_placeholder, mask)
        generated_values = self.generator(x, mask)
        final_imputed = torch.where(mask.bool(), x, generated_values)
        
        # --- CORRECCIÓN AQUÍ ---
        # Convertir el tensor booleano a float antes de la operación
        hint_noise = (torch.rand_like(x) < 0.5).float()
        hint = mask * hint_noise + 0.5 * (1 - hint_noise)
        
        # Se usa .detach() para que los gradientes del discriminador no fluyan hacia el generador
        predicted_mask = self.discriminator(final_imputed.detach(), hint)
        
        # Devolvemos final_imputed (para la predicción) y generated_values (para la pérdida de reconstrucción)
        return final_imputed, predicted_mask, generated_values

# --- Modelo Principal Integrado (Corregido) ---

class RiskAugmentedJointLSTMClassifier_TSGAIN(nn.Module):
    """
    Modelo final que ahora usa un LSTM simple después de la imputación con TSGAIN.
    """
    def __init__(self, input_size, static_indices, hidden_size, dropout_rate=0.3, num_layers=2):
        super().__init__()
        self.static_indices = static_indices
        self.dynamic_indices = [i for i in range(input_size) if i not in static_indices]
        self.input_dynamic_size = len(self.dynamic_indices)

        self.imputation_module = TSGAIN_Module(dynamic_dim=self.input_dynamic_size)

        self.f_age = nn.Sequential(MonotonicLinear(1, 1, [0]), nn.Softplus())
        self.f_apoe = nn.Sequential(MonotonicLinear(1, 1, [0]), nn.Softplus())
        self.f_edu = nn.Sequential(MonotonicLinear(1, 1, [0]), nn.Softplus())
        self.f_sex = nn.Linear(1, 1)
        self.output_risk = nn.Sigmoid()

        # --- CORRECCIÓN ARQUITECTÓNICA ---
        # Usamos un LSTM simple que no intenta hacer una segunda imputación.
        self.lstm_model = NeuroLSTM(
            input_size=self.input_dynamic_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )
        
        self.classifier = DxClassifier(input_size=self.input_dynamic_size + 1)

    def forward(self, x, mask_dyn, diag_init):
        x_static = x[:, 0, self.static_indices]
        x_dynamic = x[:, :, self.dynamic_indices]
        
        imputed_dynamic, predicted_mask, generated_values = self.imputation_module(x_dynamic, mask_dyn)

        device = x.device
        age, edu, apoe, sex = x_static[:, [0]], -x_static[:, [1]], x_static[:, [2]], x_static[:, [3]]
        diag_init = diag_init.view(-1, 1).to(device)
        mask_cn = (diag_init == 0).float()
        risk = self.f_age(age*mask_cn) + self.f_apoe(apoe*mask_cn) + self.f_edu(edu*mask_cn) + self.f_sex(sex*mask_cn)
        risk = self.output_risk(risk) * mask_cn

        # El LSTM ahora procesa directamente la secuencia ya imputada por TSGAIN
        pred_measures = self.lstm_model(imputed_dynamic)
        
        x_diagnosis = torch.cat([pred_measures, risk], dim=-1)
        pred_dx = self.classifier(x_diagnosis)
        
        # Devolvemos todo lo necesario para la pérdida colaborativa
        return pred_measures, pred_dx, predicted_mask, generated_values

    
    



    
        
# --- Función Helper para Creación del Modelo ---
def RiskAugmentedLSTM_TSGAIN_deterministic(input_size, static_indices, hidden_size, dropout_rate=0.3, num_layers=2, seed=42):
    set_seed(seed)
    model = RiskAugmentedJointLSTMClassifier_TSGAIN(
        input_size=input_size,
        static_indices=static_indices,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_layers=num_layers
    )
    initialize_weights(model)
    return model