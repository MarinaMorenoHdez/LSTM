# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:36:11 2025

@author: cplatero
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



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



def predictive_lstm_interp_risk(model, X_test, y_test, diag_init_test, scaler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    diag_init_test = torch.tensor(diag_init_test, dtype=torch.float32).to(device)
    # mask_test = torch.tensor(mask_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        # outputs = model(X_test)
        pred_meas, pred_dx = model(X_test,diag_init_test)
        # outputs_np = outputs.cpu().numpy()
        outputs_np = torch.cat([pred_meas, pred_dx], dim=1).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        # mask_np = mask_test.cpu().numpy()

    num_feat = y_test_np.shape[1]-1
    y_pred_neuro = inverse_neuro_scaling(outputs_np[:, 0:num_feat],scaler)
    y_true_neuro = inverse_neuro_scaling(y_test_np[:, 0:num_feat],scaler)

 

    # Clasificación diagnóstico (posición 4)
    # clf_mask = mask_np[:, num_feat] > 0
    # y_true = y_test_np[clf_mask, num_feat].round()
    # y_pred = (outputs_np[clf_mask, num_feat] > 0.5).astype(int)
    # y_score = outputs_np[clf_mask, num_feat]
    
    y_true = y_test_np[:, num_feat].round()
    y_pred = outputs_np[:, num_feat]

    
    return y_true_neuro, y_pred_neuro, y_true, y_pred





# def predict_full_trajectory_risk(model, x_past, diag_init_test, n_future, device='cpu'):
#     """
#     Devuelve una trayectoria futura de longitud n_future [n_future, D_out],
#     predicha autoregresivamente a partir de la última secuencia x_past [T, D_in].
#     """
#     model.eval()
#     x_past = torch.tensor(x_past, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, D_in]
#     diag_init_test = torch.tensor(diag_init_test, dtype=torch.float32).to(device)
#     outputs = []

#     with torch.no_grad():
#         for _ in range(n_future):
#             # out = model(x_past)
#             pred_meas, pred_dx = model(x_past, diag_init_test)
#             # out = torch.cat([pred_meas, pred_dx], dim=1).cpu().numpy()
#             out = torch.cat([pred_meas, pred_dx], dim=1)  # tensor aún
#             if out.dim() == 3:
#                 out = out[:, -1, :]  # [1, D_out]
#             elif out.dim() == 2:
#                 pass
#             else:
#                 raise ValueError(f"Forma inesperada de salida del modelo: {out.shape}")

#             outputs.append(out.cpu().numpy())

#             # Tomar solo D_in primeras variables (no incluir diagnóstico)
#             D_in = x_past.shape[2]
#             next_input = out[:, :D_in].unsqueeze(1)  # [1, 1, D_in]

#             # Añadir predicción como siguiente entrada
#             x_past = torch.cat([x_past, next_input], dim=1)  # [1, T+1, D_in]

#     y_pred_future = np.concatenate(outputs, axis=0)  # [n_future, D_out]
#     return y_pred_future


def predict_full_trajectory_risk(model, x_past, diag_init, n_future, static_indices, device='cpu'):
    """
    Predice la trayectoria futura de longitud `n_future` de medidas y diagnóstico
    usando un modelo que integra riesgo basal, con ventana temporal deslizante.

    Args:
        model: modelo entrenado que implementa forward(x_seq, diag_init)
        x_past: array de forma [T, D_in], secuencia observada más reciente
        diag_init: diagnóstico inicial (0: CN, 1: MCI) para el cálculo del riesgo
        n_future: número de visitas a predecir en el futuro
        static_indices: lista de índices de variables estáticas (edad, sexo, edu, APOE4)
        device: "cpu" o "cuda"

    Returns:
        y_pred_future: array de forma [n_future, D_out] con las predicciones futuras
    """
    model.eval()
    T, D_in = x_past.shape
    x_seq = torch.tensor(x_past, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, D_in]
    diag_init = torch.tensor(diag_init, dtype=torch.float32).view(1).to(device)

    # Extraer variables estáticas de la primera visita
    x_static = x_seq[:, 0, static_indices]  # [1, len(static_indices)]

    outputs = []

    with torch.no_grad():
        for _ in range(n_future):
            # Calcular riesgo basal como lo hace el modelo internamente
            age = x_static[:, [0]]
            edu = -x_static[:, [1]]
            apoe = x_static[:, [2]]
            sex = x_static[:, [3]]
            mask_cn = (diag_init == 0).float().view(-1, 1)

            age = age * mask_cn
            edu = edu * mask_cn
            apoe = apoe * mask_cn
            sex = sex * mask_cn

            risk = model.f_age(age) + model.f_apoe(apoe) + model.f_edu(edu) + model.f_sex(sex)
            risk = model.output_risk(risk) * mask_cn  # [1, 1]
            risk_seq = risk.unsqueeze(1).expand(-1, T, -1)  # [1, T, 1]

            # Quitar variables estáticas y añadir el riesgo a las dinámicas
            dynamic_mask = torch.ones(D_in, dtype=torch.bool, device=device)
            dynamic_mask[static_indices] = False
            x_dynamic = x_seq[:, :, dynamic_mask]  # [1, T, D_dyn]
            x_combined = torch.cat([x_dynamic, risk_seq], dim=-1)  # [1, T, D_dyn+1]

            # Predicción de salida (medidas + dx)
            pred_meas, pred_dx = model(x_seq, diag_init)
            out = torch.cat([pred_meas, pred_dx], dim=1)  # [1, D_out]
            outputs.append(out.cpu().numpy())

            # Preparar la siguiente entrada dinámica
            next_dynamic = out[:, :x_dynamic.shape[-1]].unsqueeze(1)  # [1, 1, D_dyn]

            # Reconstruir entrada con dinámica + riesgo, pero ventana fija (shift)
            x_dynamic = torch.cat([x_dynamic[:, 1:], next_dynamic], dim=1)  # [1, T, D_dyn]
            x_seq = torch.cat([x_dynamic, x_static.unsqueeze(1).expand(-1, T, -1)], dim=2)  # [1, T, D_in]

    y_pred_future = np.concatenate(outputs, axis=0)  # [n_future, D_out]
    return y_pred_future


def plot_subject_trajectory(y_true, y_pred, feature_names, visit_times=None):
    """
    Dibuja las trayectorias reales (puntos) y predichas (líneas) para un sujeto.
    
    y_true: [T, D] - valores clínicos reales
    y_pred: [T, D] - valores predichos por el modelo
    feature_names: lista de longitud D
    visit_times: (opcional) eje temporal, si no se proporciona se usa np.arange(T)
    """
    T, D = y_true.shape

    if visit_times is None:
        visit_times = np.arange(T)

    fig, axes = plt.subplots(D, 1, figsize=(8, 3 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for i in range(D):
        ax = axes[i]
        ax.plot(visit_times, y_pred[:, i], label="Predicción", color="tab:blue", marker="o")
        ax.scatter(visit_times, y_true[:, i], label="Valor clínico", color="tab:orange", s=40)
        ax.set_title(feature_names[i])
        ax.set_ylabel("Valor")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Visita o tiempo")
    plt.tight_layout()
    plt.show()



def plot_subject_trajectory_with_future(y_true, y_pred_past, y_pred_future, feature_names, visit_times=None):
    """
    Dibuja valores clínicos reales (y_true), predicción del pasado (y_pred_past) y
    extrapolación futura (y_pred_future) para un sujeto.
    
    y_true: [T, D] - valores reales clínicos
    y_pred_past: [T, D] - predicciones del modelo hasta la última visita observada
    y_pred_future: [n_future, D] - predicciones del modelo en visitas futuras
    feature_names: lista con los nombres de los D marcadores
    visit_times: lista u array opcional con los tiempos reales de visita (longitud T + n_future)
    """
    T, D = y_true.shape
    n_future = y_pred_future.shape[0]
    total_visits = T + n_future

    if visit_times is None:
        visit_times = np.arange(total_visits)

    fig, axes = plt.subplots(D, 1, figsize=(8, 3 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for i in range(D):
        ax = axes[i]
        ax.plot(visit_times[:T], y_pred_past[:, i], label="Predicción pasada", color="tab:blue", marker="o")
        ax.plot(visit_times[T:], y_pred_future[:, i], label="Predicción futura", color="tab:green", linestyle="--", marker="^")
        ax.scatter(visit_times[:T], y_true[:, i], label="Valor clínico", color="tab:orange", s=40)
        ax.set_title(feature_names[i])
        ax.set_ylabel("Valor")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Visita o tiempo")
    plt.tight_layout()
    plt.show()



