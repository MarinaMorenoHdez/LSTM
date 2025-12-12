# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 19:28:54 2025

@author: cplatero
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import numpy as np

from scipy.stats import pearsonr
import pandas as pd


from LSTM_aux.RiskAugmentedLSTM_deterministic import set_seed 


class JointLossWithImputation(nn.Module):
    def __init__(self, lambda_clf=0.1, lambda_mono=0.01, lambda_smooth=0.01, lambda_impute=0.5, monotonic_signs=None):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        self.lambda_clf = lambda_clf
        self.lambda_mono = lambda_mono
        self.lambda_impute = lambda_impute
        self.lambda_smooth = lambda_smooth # <-- NUEVO
        self.monotonic_signs = monotonic_signs
    
    def forward(self, pred_measures, true_measures, mask_measures,
                pred_dx, true_dx, mask_dx,
                imputed_dynamic, true_dynamic, impute_mask, patient_ids=None):
        
        # 1. Pérdida de predicción de medidas (MAE)
        mae = self.mae_loss(pred_measures, true_measures)
        mae = (mae * mask_measures).sum() / (mask_measures.sum() + 1e-6)
    
        # 2. Pérdida de clasificación de diagnóstico (BCE)
        bce = self.bce(pred_dx.squeeze(), true_dx)
        bce = (bce * mask_dx).sum() / (mask_dx.sum() + 1e-6)
        
        # 3. Pérdida de reconstrucción/imputación (MAE)
        # Solo calculamos la pérdida en los valores que eran originalmente conocidos
        imputation_loss = self.mae_loss(imputed_dynamic, true_dynamic)
        imputation_loss = (imputation_loss * impute_mask).sum() / (impute_mask.sum() + 1e-6)
    
        # 4. Penalización por monotonicidad (opcional)
        mono_penalty = 0.0
        smooth_penalty = 0.0 # <-- INICIALIZAMOS LA NUEVA PENALIZACIÓN
        
        if self.monotonic_signs is not None and patient_ids is not None:
            diffs = pred_measures[1:] - pred_measures[:-1]
            same_patient = (patient_ids[1:] == patient_ids[:-1])
            # same_patient = (patient_ids[1:, 0] == patient_ids[:-1, 0])
            if same_patient.any():
                diffs_same = diffs[same_patient]
                signs = torch.tensor(self.monotonic_signs, dtype=pred_measures.dtype, device=pred_measures.device).view(1, -1)
                penalty = torch.relu(-diffs_same * signs)
                mono_penalty = penalty.mean()
                smooth_penalty = torch.abs(diffs_same).mean()
    
        total_loss = mae + self.lambda_clf * bce + self.lambda_impute * imputation_loss + self.lambda_mono * mono_penalty  + self.lambda_smooth * smooth_penalty
        return total_loss


def train_imputation_augmented_model(model, X_train, y_train, mask_y_train, mask_dyn_train, ID_train, diag_init_train,
                                     X_val, y_val, mask_y_val, mask_dyn_val, ID_val, diag_init_val,
                                     num_epochs=100, lr=0.001, patience=10,
                                     lambda_clf=0.1, lambda_impute=0.5, lambda_mono=0.01, lambda_smooth=0.01, monotonic_signs=None, seed=42):
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configurar el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = JointLossWithImputation(lambda_clf=lambda_clf, lambda_impute=lambda_impute, 
                                        lambda_mono=lambda_mono, lambda_smooth=lambda_smooth, monotonic_signs=monotonic_signs)
    
    # Convertir datos a tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    mask_y_train_t = torch.tensor(mask_y_train, dtype=torch.float32).to(device)
    mask_dyn_train_t = torch.tensor(mask_dyn_train, dtype=torch.float32).to(device)
    ID_train_t = torch.tensor(ID_train, dtype=torch.long).squeeze().to(device)
    diag_init_train_t = torch.tensor(diag_init_train, dtype=torch.float32).to(device)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    mask_y_val_t = torch.tensor(mask_y_val, dtype=torch.float32).to(device)
    mask_dyn_val_t = torch.tensor(mask_dyn_val, dtype=torch.float32).to(device)
    ID_val_t = torch.tensor(ID_val, dtype=torch.long).squeeze().to(device)
    diag_init_val_t = torch.tensor(diag_init_val, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    num_regression_features = y_train.shape[1] - 1
    dynamic_indices = model.dynamic_indices

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        pred_meas, pred_dx, imputed_dyn = model(X_train_t, mask_dyn_train_t, diag_init_train_t)
        
        true_dynamic_train = X_train_t[:, :, dynamic_indices].reshape(-1, len(dynamic_indices))
        imputed_dyn_flat = imputed_dyn.reshape(-1, len(dynamic_indices))
        mask_dyn_train_flat = mask_dyn_train_t.reshape(-1, len(dynamic_indices))

        loss = criterion(
            pred_meas, y_train_t[:, :num_regression_features], mask_y_train_t[:, :num_regression_features],
            pred_dx, y_train_t[:, -1], mask_y_train_t[:, -1],
            imputed_dyn_flat, true_dynamic_train, mask_dyn_train_flat,
            patient_ids=ID_train_t
        )
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_meas, val_dx, val_imputed_dyn = model(X_val_t, mask_dyn_val_t, diag_init_val_t)
            
            true_dynamic_val = X_val_t[:, :, dynamic_indices].reshape(-1, len(dynamic_indices))
            val_imputed_dyn_flat = val_imputed_dyn.reshape(-1, len(dynamic_indices))
            mask_dyn_val_flat = mask_dyn_val_t.reshape(-1, len(dynamic_indices))

            val_loss = criterion(
                val_meas, y_val_t[:, :num_regression_features], mask_y_val_t[:, :num_regression_features],
                val_dx, y_val_t[:, -1], mask_y_val_t[:, -1],
                val_imputed_dyn_flat, true_dynamic_val, mask_dyn_val_flat,
                patient_ids=ID_val_t
            ).item()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


import optuna

def train_imputation_augmented_model_logged(model, 
                                            X_train, y_train, mask_y_train, mask_dyn_train, ID_train, diag_init_train,
                                            X_val, y_val, mask_y_val, mask_dyn_val, ID_val, diag_init_val,
                                            num_epochs=100, lr=0.001, patience=10,
                                            lambda_clf=0.1, lambda_impute=0.5, lambda_mono=0.01, lambda_smooth=0.01,
                                            monotonic_signs=None, seed=42):
    """
    Versión con logging de todas las componentes de pérdida
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = JointLossWithImputation(lambda_clf=lambda_clf, lambda_impute=lambda_impute, 
                                        lambda_mono=lambda_mono, lambda_smooth=lambda_smooth, 
                                        monotonic_signs=monotonic_signs)

    # Tensores
    def to_torch(arr, dtype=torch.float32): return torch.tensor(arr, dtype=dtype).to(device)
    X_train_t, y_train_t = to_torch(X_train), to_torch(y_train)
    mask_y_train_t, mask_dyn_train_t = to_torch(mask_y_train), to_torch(mask_dyn_train)
    ID_train_t, diag_init_train_t = to_torch(ID_train, torch.long).squeeze(), to_torch(diag_init_train)

    X_val_t, y_val_t = to_torch(X_val), to_torch(y_val)
    mask_y_val_t, mask_dyn_val_t = to_torch(mask_y_val), to_torch(mask_dyn_val)
    ID_val_t, diag_init_val_t = to_torch(ID_val, torch.long).squeeze(), to_torch(diag_init_val)

    num_regression_features = y_train.shape[1] - 1
    dynamic_indices = model.dynamic_indices

    best_val_loss, best_state = float("inf"), None
    epochs_no_improve = 0

    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred_meas, pred_dx, imputed_dyn = model(X_train_t, mask_dyn_train_t, diag_init_train_t)

        true_dyn_train = X_train_t[:, :, dynamic_indices].reshape(-1, len(dynamic_indices))
        imputed_dyn_flat = imputed_dyn.reshape(-1, len(dynamic_indices))
        mask_dyn_flat = mask_dyn_train_t.reshape(-1, len(dynamic_indices))

        # Forward + loss
        loss = criterion(pred_meas, y_train_t[:, :num_regression_features], mask_y_train_t[:, :num_regression_features],
                         pred_dx, y_train_t[:, -1], mask_y_train_t[:, -1],
                         imputed_dyn_flat, true_dyn_train, mask_dyn_flat,
                         patient_ids=ID_train_t)

        loss.backward()
        optimizer.step()

        # Evaluación en validación
        model.eval()
        with torch.no_grad():
            val_meas, val_dx, val_imputed_dyn = model(X_val_t, mask_dyn_val_t, diag_init_val_t)
            true_dyn_val = X_val_t[:, :, dynamic_indices].reshape(-1, len(dynamic_indices))
            val_imputed_dyn_flat = val_imputed_dyn.reshape(-1, len(dynamic_indices))
            mask_dyn_val_flat = mask_dyn_val_t.reshape(-1, len(dynamic_indices))

            val_loss = criterion(val_meas, y_val_t[:, :num_regression_features], mask_y_val_t[:, :num_regression_features],
                                 val_dx, y_val_t[:, -1], mask_y_val_t[:, -1],
                                 val_imputed_dyn_flat, true_dyn_val, mask_dyn_val_flat,
                                 patient_ids=ID_val_t)

        # Guardamos pérdidas
        history["train"].append(loss.item())
        history["val"].append(val_loss.item())

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping en epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


# ---------------------------
# Función objetivo para Optuna
# ---------------------------

# def objective(trial, model_class, data, monotonic_signs=None, seed=42):
#     """
#     data = (X_train, y_train, mask_y_train, mask_dyn_train, ID_train, diag_init_train,
#             X_val, y_val, mask_y_val, mask_dyn_val, ID_val, diag_init_val)
#     """

#     lambda_clf = trial.suggest_categorical("lambda_clf", [0.1, 0.2, 0.5])
#     lambda_impute = trial.suggest_categorical("lambda_impute", [0.5, 1.0, 2.0, 5.0])
#     lambda_mono = trial.suggest_categorical("lambda_mono", [0.1, 0.5, 1.0, 2.0])
#     lambda_smooth = trial.suggest_categorical("lambda_smooth", [0.01, 0.05, 0.1, 0.5])

#     # Instanciamos modelo
#     X_train, y_train, mask_y_train, mask_dyn_train, ID_train, diag_init_train, \
#     X_val, y_val, mask_y_val, mask_dyn_val, ID_val, diag_init_val = data

#     model = model_class(input_size=X_train.shape[2],
#                         static_indices=[0],  # <- O ajustar según tu dataset
#                         hidden_size=64,
#                         dropout_rate=0.3,
#                         num_layers=2)

#     model, history = train_imputation_augmented_model_logged(
#         model,
#         X_train, y_train, mask_y_train, mask_dyn_train, ID_train, diag_init_train,
#         X_val, y_val, mask_y_val, mask_dyn_val, ID_val, diag_init_val,
#         num_epochs=100, lr=0.001, patience=10,
#         lambda_clf=lambda_clf, lambda_impute=lambda_impute,
#         lambda_mono=lambda_mono, lambda_smooth=lambda_smooth,
#         monotonic_signs=monotonic_signs, seed=seed
#     )

#     # Retornar el mejor val_loss
#     return min(history["val"])


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

def evaluate_imputation_augmented_model(model, X_test, y_test, mask_y_test, mask_dyn_test, ID_test, diag_init_test, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convertir datos a tensores
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    mask_dyn_test_t = torch.tensor(mask_dyn_test, dtype=torch.float32).to(device)
    diag_init_test_t = torch.tensor(diag_init_test, dtype=torch.float32).to(device)
    
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    mask_y_test = torch.tensor(mask_y_test, dtype=torch.float32).to(device)
 
    with torch.no_grad():
        pred_meas, pred_dx, _ = model(X_test_t, mask_dyn_test_t, diag_init_test_t)
    
    # # Mover resultados a la CPU y convertir a NumPy
    # pred_meas_np = pred_meas.cpu().numpy()
    # pred_dx_np = pred_dx.cpu().numpy()
    
    # Aquí puedes seguir la misma lógica de tu función `evaluate_risk_augmented_model`
    # para calcular MAE, Accuracy, AUC, etc., usando pred_meas_np y pred_dx_np.
    # El resto del código de evaluación de LSTM_risk_interp_determ_incr.py sería aplicable aquí.
    # (El código de la función `evaluate_risk_augmented_model` se puede reutilizar casi directamente)
 

    # Move predictions and true labels to CPU for NumPy operations
    outputs_np = torch.cat([pred_meas, pred_dx], dim=1).cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    mask_np = mask_y_test.cpu().numpy()
 
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
    sens = recall_score(y_true_dx, y_pred_dx_binary)
    spec = recall_score(y_true_dx, y_pred_dx_binary, pos_label=0)
    auc = roc_auc_score(y_true_dx, y_pred_dx_score)
    f1 = f1_score(y_true_dx, y_pred_dx_binary)
    bacc = balanced_accuracy_score(y_true_dx, y_pred_dx_binary)
    
    print({
        "MAE_mean": maes,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "AUC": auc
    })

    ID = ID_test[mask_dx, :]
    clinic_data = np.column_stack((ID, y_true_dx.astype(int), y_pred_dx_binary))
    pCU_idx_onset = clinical_scores(clinic_data)
 
    # return {
    #     'mae_per_variable': maes,
    #     'accuracy': acc,
    #     'sensitivity': sens,
    #     'specificity': spec,
    #     'auc': auc,
    #     'f1': f1,
    #     'balanced_accuracy': bacc
    # }
 
    
    
 