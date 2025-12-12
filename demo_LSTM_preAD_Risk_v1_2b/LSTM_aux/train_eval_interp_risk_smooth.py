# -*- coding: utf-8 -*-
"""
Created on Fri May 30 15:59:29 2025

@author: cplatero
"""

# -*- coding: utf-8 -*-
"""
Red integrada LSTM + riesgo basal (solo si el diagnóstico inicial es CN).
No se realiza imputación de datos faltantes porque las secuencias están completas.
Se asume que las entradas son las medidas neuropsicológicas dinámicas y que las variables
estáticas (edad, educación, APOE4, sexo) se usan para calcular el riesgo basal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import numpy as np

from scipy.stats import pearsonr
import pandas as pd

import random
import os


    
class JointLoss(nn.Module):
    def __init__(self, lambda_clf=0.1, lambda_mono=0.01, lambda_smooth=0.01, monotonic_signs=None):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.lambda_clf = lambda_clf
        self.lambda_mono = lambda_mono
        self.monotonic_signs = monotonic_signs  # lista con +1, -1 o 0 por variable
        self.lambda_smooth = lambda_smooth # <-- NUEVO

    def forward(self, pred_measures, true_measures, mask_measures,
                      pred_dx, true_dx, mask_dx, patient_ids=None):

        mae = torch.abs(pred_measures - true_measures)
        mae = (mae * mask_measures).sum() / (mask_measures.sum() + 1e-6)

        bce = self.bce(pred_dx.squeeze(), true_dx)
        bce = (bce * mask_dx).sum() / (mask_dx.sum() + 1e-6)

        mono_penalty = 0.0
        smooth_penalty = 0.0 # <-- INICIALIZAMOS LA NUEVA PENALIZACIÓN
        
        # if self.monotonic_signs is not None and patient_ids is not None:
        #     diffs = pred_measures[1:] - pred_measures[:-1]  # [B-1, D]
        #     same_patient = (patient_ids[1:] == patient_ids[:-1])  # [B-1]
        #     if same_patient.any():
        #         diffs_same = diffs[same_patient]
        #         signs = torch.tensor(self.monotonic_signs, dtype=pred_measures.dtype, device=pred_measures.device).view(1, -1)
        #         penalty = torch.relu(-diffs_same * signs)
        #         mono_penalty = penalty.mean()
                
        if patient_ids is not None:
            # Diferencias entre predicciones de pasos de tiempo consecutivos
            diffs = pred_measures[1:] - pred_measures[:-1]
            # Máscara para asegurar que las diferencias se calculan solo para el mismo paciente
            same_patient = (patient_ids[1:] == patient_ids[:-1])
            
            if same_patient.any():
                diffs_same_patient = diffs[same_patient]

                # 5. Penalización por monotonicidad (sin cambios)
                if self.monotonic_signs is not None and self.lambda_mono > 0:
                    signs = torch.tensor(self.monotonic_signs, dtype=pred_measures.dtype, device=pred_measures.device).view(1, -1)
                    penalty = torch.relu(-diffs_same_patient * signs)
                    mono_penalty = penalty.mean()

                # 6. NUEVA PENALIZACIÓN POR SUAVIDAD (L1 de las diferencias)
                if self.lambda_smooth > 0:
                    # Penalizamos la magnitud absoluta de los cambios
                    smooth_penalty = torch.abs(diffs_same_patient).mean()

        return mae + self.lambda_clf * bce + self.lambda_mono * mono_penalty + self.lambda_smooth * smooth_penalty    



def train_risk_augmented_model_smooth(model, X_train, y_train, mask_y_train, ID_train, diag_init_train,
                               X_val, y_val, mask_y_val, ID_val, diag_init_val,
                               num_epochs=1000, lr=0.001, patience=10,
                               lambda_clf=0.1, lambda_mono=0.01, lambda_smooth=0.01, monotonic_signs=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_regression_features = y_train.shape[1] - 1

    criterion = JointLoss(lambda_clf=lambda_clf, lambda_mono=lambda_mono, lambda_smooth=lambda_smooth, monotonic_signs=monotonic_signs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convertir a tensores
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    mask_y_train = torch.tensor(mask_y_train, dtype=torch.float32).to(device)
    ID_train = torch.tensor(ID_train, dtype=torch.long).squeeze().to(device)
    diag_init_train = torch.tensor(diag_init_train, dtype=torch.float32).to(device)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    mask_y_val = torch.tensor(mask_y_val, dtype=torch.float32).to(device)
    ID_val = torch.tensor(ID_val, dtype=torch.long).squeeze().to(device)
    diag_init_val = torch.tensor(diag_init_val, dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        pred_meas, pred_dx = model(X_train, diag_init_train)

        loss = criterion(
            pred_meas, y_train[:, :num_regression_features], mask_y_train[:, :num_regression_features],
            pred_dx, y_train[:, -1], mask_y_train[:, -1], patient_ids=ID_train
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_meas, val_dx = model(X_val, diag_init_val)
            val_loss = criterion(
                val_meas, y_val[:, :num_regression_features], mask_y_val[:, :num_regression_features],
                val_dx, y_val[:, -1], mask_y_val[:, -1], patient_ids=ID_val
            ).item()

        print(f"Epoch {epoch+1} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model


# -----------------------------
# 5. Evaluación
# -----------------------------
def clinical_scores(data):
    # Supongamos que `data` es tu matriz numpy con las columnas:
    # [ID_paciente, orden_visita, diagnostico_clinico, diagnostico_IA]
    # Convertimos a DataFrame para facilitar el análisis:
    columns = ['ID', 'visit_order', 'clin_dx', 'ia_dx']
    df = pd.DataFrame(data, columns=columns)
    
    # Agrupar por paciente
    grouped = df.groupby('ID')
    
    # Inicializamos contadores
    estables_sanos = 0
    estables_sanos_detectados = 0
    
    estables_enfermos = 0
    estables_enfermos_detectados = 0
    
    progresores = 0
    progresores_detectados = 0
    primeras_detectadas = []
    
    # Analizamos cada paciente
    for pid, group in grouped:
        group_sorted = group.sort_values('visit_order')
        clinic_value =  group_sorted['clin_dx'].values
        clin_inicio = group_sorted['clin_dx'].iloc[0]
        clin_fin = group_sorted['clin_dx'].iloc[-1]
        ia_diag = group_sorted['ia_dx'].values
        ia_diag_fin = group_sorted['ia_dx'].iloc[-1]
    
        if clin_inicio == 0 and clin_fin == 0:
            # Estable sano
            estables_sanos += 1
            if np.all(ia_diag_fin == 0):
                estables_sanos_detectados += 1
    
        elif clin_inicio == 1 and clin_fin == 1:
            # Estable enfermo
            estables_enfermos += 1
            if np.all(ia_diag_fin == 1):
                estables_enfermos_detectados += 1
    
        elif clin_inicio == 0 and clin_fin == 1:
            # Progresor
            progresores += 1
            # Ver si la IA en algún momento predijo 1
            if  ia_diag_fin == 1:
                progresores_detectados += 1
                first_clinic = group_sorted['visit_order'].iloc[np.argmax(clinic_value == 1)]
                first_detected = group_sorted['visit_order'].iloc[np.argmax(ia_diag == 1)]
                primeras_detectadas.append((pid, first_clinic, first_detected))
    
    # Cálculo de porcentajes
    porc_estables_sanos = 100 * estables_sanos_detectados / estables_sanos if estables_sanos > 0 else 0
    porc_estables_enfermos = 100 * estables_enfermos_detectados / estables_enfermos if estables_enfermos > 0 else 0
    porc_progresores = 100 * progresores_detectados / progresores if progresores > 0 else 0
    
    print(f"sCU: {estables_sanos}, detectados correctamente: {estables_sanos_detectados} ({porc_estables_sanos:.1f}%)")
    print(f"sMCI: {estables_enfermos}, detectados correctamente: {estables_enfermos_detectados} ({porc_estables_enfermos:.1f}%)")
    print(f"pCU: {progresores}, detectados por la IA: {progresores_detectados} ({porc_progresores:.1f}%)")

    # Correlación de Pearson entre visitas de detección
    if primeras_detectadas:
        visitas_ia = [v[2] for v in primeras_detectadas]
        visitas_clin = [v[1] for v in primeras_detectadas]
        corr, pval = pearsonr(visitas_ia, visitas_clin)
        print(f"Correlación de Pearson entre detección IA y clínica: r = {corr:.3f}, p = {pval:.3e}")
    else:
        print("No se detectaron progresores correctamente por la IA. No se puede calcular correlación.")
    
      
    return primeras_detectadas
    


def inverse_neuro_scaling(y, scaler):
    """
    Aplica la transformación inversa del StandardScaler SOLO a las primeras 4 columnas
    de medidas neuropsicológicas.
    """
    # Crear una copia de los datos con tantas columnas como tenía el scaler
    padded = np.zeros((y.shape[0], scaler.mean_.shape[0]))
    num_feat=y.shape[1]
    padded[:, :num_feat] = y  # Las 4 medidas neuropsicológicas están en las primeras columnas

    # Aplicar transformación inversa global
    inv = scaler.inverse_transform(padded)

    # Extraer solo las 4 primeras columnas
    return inv[:, :num_feat]


 
def evaluate_risk_augmented_model(model, X_test, y_test, mask_y_test, ID_test, diag_init_test, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    mask_y_test = torch.tensor(mask_y_test, dtype=torch.float32).to(device)
    diag_init_test = torch.tensor(diag_init_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_meas, pred_dx = model(X_test, diag_init_test)

    outputs_np = torch.cat([pred_meas, pred_dx], dim=1).cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    mask_np = mask_y_test.cpu().numpy()

    num_feat = y_test_np.shape[1] - 1
    y_pred_neuro = inverse_neuro_scaling(outputs_np[:, :num_feat], scaler)
    y_true_neuro = inverse_neuro_scaling(y_test_np[:, :num_feat], scaler)

    # MAE
    mae_list = []
    for i in range(num_feat):
        valid_idx = mask_np[:, i] > 0
        mae_i = np.abs(y_pred_neuro[valid_idx, i] - y_true_neuro[valid_idx, i]).mean()
        mae_list.append(mae_i)

    # Diagnóstico
    clf_mask = mask_np[:, num_feat] > 0
    y_true = y_test_np[clf_mask, num_feat].round()
    y_pred = (outputs_np[clf_mask, num_feat] > 0.5).astype(int)
    y_score = outputs_np[clf_mask, num_feat]

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    spec = recall_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_score)

    print({
        "MAE_mean": mae_list,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "AUC": auc
    })

    ID = ID_test[clf_mask, :]
    clinic_data = np.column_stack((ID, y_true.astype(int), y_pred))
    pCU_idx_onset = clinical_scores(clinic_data)

   
