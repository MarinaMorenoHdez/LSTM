# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 19:24:18 2025

@author: cplatero
"""

import torch
import torch.nn as nn


# Asumiendo que edcf_module.py y LSTM_risk_interp_determ_incr.py están en el mismo directorio
# y sus clases (EDCFModule, RiskAugmentedJointLSTMClassifier, etc.) están disponibles.
from LSTM_aux.edcf_module import EDCFModule
from LSTM_aux.RiskAugmentedLSTM_deterministic import RiskAugmentedJointLSTMClassifier, initialize_weights

class ImputationAugmentedLSTM(nn.Module):
    def __init__(self, input_size, static_indices, hidden_size, dropout_rate=0.3, num_layers=2):
        super().__init__()
        
        self.static_indices = static_indices
        self.dynamic_indices = [i for i in range(input_size) if i not in static_indices]
        self.input_dynamic_size = len(self.dynamic_indices)

        # Módulo para imputar las variables dinámicas
        self.imputation_module = EDCFModule(input_dim=self.input_dynamic_size)
        
        # Modelo principal de predicción
        self.prediction_module = RiskAugmentedJointLSTMClassifier(
            input_size=input_size,
            static_indices=static_indices,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )
        
        initialize_weights(self)

    def forward(self, x, x_mask, diag_init):
        # x: [B, T, D], x_mask: [B, T, D_dyn]
        
        x_dynamic = x[:, :, self.dynamic_indices]
        
        # 1. Imputar valores faltantes en las variables dinámicas
        # El EDCFModule espera una entrada de [B, input_dim].
        # Lo aplicaremos a cada paso de tiempo.
        B, T, D_dyn = x_dynamic.shape
        
        # Aplanar para procesar con EDCF
        x_dynamic_flat = x_dynamic.reshape(B * T, D_dyn)
        x_mask_flat = x_mask.reshape(B * T, D_dyn)
        
        # Crear una versión de los datos donde los valores faltantes son 0
        x_dynamic_masked_input = x_dynamic_flat * x_mask_flat
        # x_dynamic_masked_input = x_dynamic_flat 
        
        # El módulo EDCF no usa explícitamente la máscara en su forward,
        # pero opera sobre los datos donde los faltantes fueron puestos a cero.
        # En esta versión, ignoramos x_cat como se indica en el fichero edcf_module.py
        imputed_dynamic_flat = self.imputation_module(x_dynamic_masked_input, x_cat=None)
        
        # Usar los valores imputados solo donde faltaban los originales
        final_dynamic_flat = torch.where(x_mask_flat.bool(), x_dynamic_flat, imputed_dynamic_flat)
        final_dynamic = final_dynamic_flat.reshape(B, T, D_dyn)

        # 2. Reconstruir la secuencia completa con los datos imputados
        x_imputed = x.clone()
        x_imputed[:, :, self.dynamic_indices] = final_dynamic
        
        # 3. Pasar la secuencia completa al modelo de predicción
        pred_measures, pred_dx = self.prediction_module(x_imputed, diag_init)
        
        return pred_measures, pred_dx, final_dynamic
    
    
