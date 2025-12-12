# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 13:26:51 2025

@author: cplatero
"""

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import numpy as np

from scipy.stats import pearsonr
import pandas as pd


from LSTM_aux.RiskAugmentedLSTM_deterministic import set_seed 

class TSGAINJointLoss(nn.Module):
    """
    Pérdida unificada para el generador de TSGAIN y las tareas posteriores.
    """
    # AÑADIMOS lambda_smooth AL CONSTRUCTOR
    def __init__(self, lambda_impute=0.5, lambda_clf=0.1, lambda_mono=0.01, 
                 lambda_smooth=0.01, monotonic_signs=None): # <-- NUEVO PARÁMETRO
        super().__init__()
        self.lambda_impute = lambda_impute
        self.lambda_clf = lambda_clf
        self.lambda_mono = lambda_mono
        self.lambda_smooth = lambda_smooth # <-- NUEVO
        self.monotonic_signs = monotonic_signs
        
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, 
                # ... (parámetros sin cambios)
                pred_measures, true_measures, mask_measures,
                pred_dx, true_dx, mask_dx,
                predicted_mask_from_G,
                generated_values, true_dynamic, mask_dynamic,
                patient_ids=None):

        # ... (Cálculos de loss_g_adv, recon_loss, loss_mae_pred, loss_bce_pred sin cambios)
        # 1. Pérdida adversaria del generador
        loss_g_adv = -torch.log(predicted_mask_from_G + 1e-8).mean()
        
        # 2. Pérdida de reconstrucción (imputación)
        recon_loss_unreduced = self.mae_loss(generated_values, true_dynamic)
        recon_loss = (recon_loss_unreduced * mask_dynamic).sum() / (mask_dynamic.sum() + 1e-6)

        # 3. Pérdida de predicción de medidas (MAE)
        mae_pred_unreduced = self.mae_loss(pred_measures, true_measures)
        loss_mae_pred = (mae_pred_unreduced * mask_measures).sum() / (mask_measures.sum() + 1e-6)
        
        # 4. Pérdida de clasificación de diagnóstico (BCE)
        bce_pred_unreduced = self.bce_loss(pred_dx.squeeze(), true_dx)
        loss_bce_pred = (bce_pred_unreduced * mask_dx).sum() / (mask_dx.sum() + 1e-6)
        
        mono_penalty = 0.0
        smooth_penalty = 0.0 # <-- INICIALIZAMOS LA NUEVA PENALIZACIÓN
        
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
        
        # Pérdida total ponderada (INCLUIMOS LA NUEVA PENALIZACIÓN)
        total_loss = (loss_g_adv + 
                      self.lambda_impute * recon_loss + 
                      loss_mae_pred + 
                      self.lambda_clf * loss_bce_pred + 
                      self.lambda_mono * mono_penalty +
                      self.lambda_smooth * smooth_penalty) # <-- NUEVO
                      
        return total_loss
# =============================================================================
# 2. FUNCIÓN DE ENTRENAMIENTO MODIFICADA
# =============================================================================
def train_tsgain_model_smooth(model, X_train, y_train, mask_y_train, mask_dyn_train, ID_train, diag_init_train,
                       X_val, y_val, mask_y_val, mask_dyn_val, ID_val, diag_init_val,
                       num_epochs=100, lr=0.001, patience=10, lambda_impute=0.5, lambda_clf=0.1,
                       lambda_mono=0.01, lambda_smooth=0.01,monotonic_signs=None,
                       seed=42):
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Separar parámetros para los dos optimizadores
    d_params = model.imputation_module.discriminator.parameters()
    g_params = [p for n, p in model.named_parameters() if 'discriminator' not in n]
    optimizer_d = optim.Adam(d_params, lr=lr)
    optimizer_g = optim.Adam(g_params, lr=lr)
    
    criterion_d = nn.BCELoss()
    criterion_g = TSGAINJointLoss(
        lambda_impute=lambda_impute,
        lambda_clf=lambda_clf,
        lambda_mono=lambda_mono,
        lambda_smooth=lambda_smooth,
        monotonic_signs=monotonic_signs
    )
    val_mae_loss = nn.L1Loss()
    val_bce_loss = nn.BCELoss()
    
    # d_params = model.imputation_module.discriminator.parameters()
    # g_params = [p for n, p in model.named_parameters() if 'discriminator' not in n]
    
    # optimizer_d = optim.Adam(d_params, lr=lr)
    # optimizer_g = optim.Adam(g_params, lr=lr)
    
    # bce_loss = nn.BCELoss()
    # mae_loss = nn.L1Loss()
    
    # --- Convertir datos a tensores ---
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    mask_y_train_t = torch.tensor(mask_y_train, dtype=torch.float32).to(device)
    mask_dyn_train_t = torch.tensor(mask_dyn_train, dtype=torch.float32).to(device)
    diag_init_train_t = torch.tensor(diag_init_train, dtype=torch.float32).to(device)
    ID_train_t = torch.tensor(ID_train, dtype=torch.long).squeeze().to(device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    mask_y_val_t = torch.tensor(mask_y_val, dtype=torch.float32).to(device)
    mask_dyn_val_t = torch.tensor(mask_dyn_val, dtype=torch.float32).to(device)
    diag_init_val_t = torch.tensor(diag_init_val, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    num_regression_features = y_train.shape[1] - 1
    dynamic_indices = model.dynamic_indices

    for epoch in range(num_epochs):
        model.train()
        
        # --- PASO 1: Entrenar el Discriminador ---
        optimizer_d.zero_grad()
        _, _, predicted_mask, _ = model(X_train_t, mask_dyn_train_t, diag_init_train_t)
        # loss_d = bce_loss(predicted_mask, mask_dyn_train_t)
        loss_d = criterion_d(predicted_mask, mask_dyn_train_t)
        loss_d.backward()
        optimizer_d.step()

        
        # --- PASO 2: Entrenar el Generador y el resto del modelo ---
        optimizer_g.zero_grad()
        # pred_meas, pred_dx, predicted_mask_g, generated_values = model(X_train_t, mask_dyn_train_t)
        pred_meas, pred_dx, predicted_mask_g, generated_values = model(X_train_t, mask_dyn_train_t, diag_init_train_t)
        
        true_dynamic_train = X_train_t[:, :, dynamic_indices]
        
        total_loss_g = criterion_g(
            pred_meas, y_train_t[:, :num_regression_features], mask_y_train_t[:, :num_regression_features],
            pred_dx, y_train_t[:, -1], mask_y_train_t[:, -1],
            predicted_mask_g,
            generated_values, true_dynamic_train, mask_dyn_train_t,
            patient_ids=ID_train_t
        )
        
        total_loss_g.backward()
        optimizer_g.step()

        model.eval()
        
        # optimizer_g.zero_grad()
        # pred_meas, pred_dx, predicted_mask, generated_values = model(X_train_t, mask_dyn_train_t, diag_init_train_t)
        
        # loss_g_adv = -torch.log(predicted_mask + 1e-8).mean() # Equivalente a la pérdida del generador en la literatura
        
        # true_dynamic_train = X_train_t[:, :, dynamic_indices]
        # recon_loss = mae_loss(generated_values[mask_dyn_train_t.bool()], true_dynamic_train[mask_dyn_train_t.bool()])
        
        # loss_mae_pred = mae_loss(pred_meas[mask_y_train_t[:, :num_regression_features].bool()], 
        #                          y_train_t[:, :num_regression_features][mask_y_train_t[:, :num_regression_features].bool()])
                                 
        # loss_bce_pred = bce_loss(pred_dx.squeeze()[mask_y_train_t[:, -1].bool()], 
        #                          y_train_t[:, -1][mask_y_train_t[:, -1].bool()])
        
        # total_loss_g = loss_g_adv + lambda_impute * recon_loss + loss_mae_pred + lambda_clf * loss_bce_pred
        # total_loss_g.backward()
        # optimizer_g.step()

        # # --- Validación ---
        # model.eval()
        with torch.no_grad():
            val_meas, val_dx, _, _ = model(X_val_t, mask_dyn_val_t, diag_init_val_t)
            valid_mask_meas = mask_y_val_t[:, :num_regression_features].bool()
            valid_mask_dx = mask_y_val_t[:, -1].bool()
            
            if valid_mask_meas.sum() > 0 and valid_mask_dx.sum() > 0:
                val_loss_mae = val_mae_loss(val_meas[valid_mask_meas], 
                                            y_val_t[:, :num_regression_features][valid_mask_meas])
                val_loss_bce = val_bce_loss(val_dx.squeeze()[valid_mask_dx], 
                                            y_val_t[:, -1][valid_mask_dx])
                val_loss = val_loss_mae + lambda_clf * val_loss_bce
            else:
                val_loss = torch.tensor(float('inf'))

            # val_meas, val_dx, _, _ = model(X_val_t, mask_dyn_val_t, diag_init_val_t)
            # val_loss_mae = mae_loss(val_meas[mask_y_val_t[:, :num_regression_features].bool()],
            #                         y_val_t[:, :num_regression_features][mask_y_val_t[:, :num_regression_features].bool()])
            # val_loss_bce = bce_loss(val_dx.squeeze()[mask_y_val_t[:, -1].bool()],
            #                         y_val_t[:, -1][mask_y_val_t[:, -1].bool()])
            # val_loss = val_loss_mae + lambda_clf * val_loss_bce

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss G: {total_loss_g.item():.4f} | Train Loss D: {loss_d.item():.4f} | Val Loss: {val_loss.item():.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Parada temprana en epoch {epoch+1}")
                break
    
    if best_model_state:
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



from sklearn.metrics import (
    mean_absolute_error, accuracy_score, recall_score, roc_auc_score,
    f1_score, balanced_accuracy_score
)

# --- Función de evaluación (mismo patrón que el entrenamiento) ---
def evaluate_tsgain_model(model, X_test, y_test, mask_y_test, mask_dyn_test, ID_test, diag_init_test, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    mask_y_test_t = torch.tensor(mask_y_test, dtype=torch.float32).to(device)
    mask_dyn_test_t = torch.tensor(mask_dyn_test, dtype=torch.float32).to(device)
    diag_init_test_t = torch.tensor(diag_init_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_meas, pred_dx, _, _ = model(X_test_t, mask_dyn_test_t, diag_init_test_t)
        
    outputs_np = torch.cat([pred_meas, pred_dx], dim=1).cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()
    mask_np = mask_y_test_t.cpu().numpy()
    
    num_feat = y_test_np.shape[1] - 1
    y_pred_meas = inverse_neuro_scaling(outputs_np[:, :num_feat], scaler)
    y_true_meas = inverse_neuro_scaling(y_test_np[:, :num_feat], scaler)

    maes = []
    for j in range(num_feat):
        mask_j = mask_np[:, j] > 0
        if mask_j.sum() > 0:
            maes.append(mean_absolute_error(y_true_meas[mask_j, j], y_pred_meas[mask_j, j]))
        else:
            maes.append(np.nan)
    
    mask_dx = mask_np[:, -1] > 0
    y_true_dx = y_test_np[mask_dx, -1].round().astype(int)
    y_pred_dx_binary = (outputs_np[mask_dx, -1] > 0.5).astype(int)
    y_pred_dx_score = outputs_np[mask_dx, -1]
    
    acc = accuracy_score(y_true_dx, y_pred_dx_binary)
    sens = recall_score(y_true_dx, y_pred_dx_binary, pos_label=1)
    spec = recall_score(y_true_dx, y_pred_dx_binary, pos_label=0)
    auc = roc_auc_score(y_true_dx, y_pred_dx_score)

    print("\n--- Resultados de Evaluación ---")
    print(f"MAE por Medida: {maes}")
    print(f"Accuracy: {acc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, AUC: {auc:.4f}")
    
    ID = ID_test[mask_dx, :]
    clinic_data = np.column_stack((ID, y_true_dx, y_pred_dx_binary))
    clinical_scores(clinic_data) # Asegúrate de que esta función maneje el formato correcto
