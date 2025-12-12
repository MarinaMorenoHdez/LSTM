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



def predictive_lstm_TSGAIN_risk(model, X_test, mask_dyn_test, y_test, diag_init_test, scaler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convertir datos a tensores
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    mask_dyn_test_t = torch.tensor(mask_dyn_test, dtype=torch.float32).to(device)
    diag_init_test_t = torch.tensor(diag_init_test, dtype=torch.float32).to(device)
    
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
 
    with torch.no_grad():
        pred_meas, pred_dx, _, _ = model(X_test_t, mask_dyn_test_t, diag_init_test_t)
        outputs_np = torch.cat([pred_meas, pred_dx], dim=1).cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    num_feat = y_test_np.shape[1]-1
    y_pred_neuro = inverse_neuro_scaling(outputs_np[:, 0:num_feat],scaler)
    y_true_neuro = inverse_neuro_scaling(y_test_np[:, 0:num_feat],scaler)

 

    
    y_true = y_test_np[:, num_feat].round()
    y_pred = outputs_np[:, num_feat]

    
    return y_true_neuro, y_pred_neuro, y_true, y_pred








def predict_full_trajectory_risk(model, x_past, mask_past, diag_init, n_future, static_indices, scaler=None, device='cpu'):
    """
    Versión corregida y robusta que predice la trayectoria futura de forma autorregresiva.
    Maneja correctamente las dimensiones y la evolución de las secuencias y las máscaras.
    """
    model.eval()
    model.to(device)

    # 1. Separar las partes estáticas y dinámicas de la secuencia inicial
    T, D_in = x_past.shape
    _, D_in_dyn = mask_past.shape
    dynamic_indices = [i for i in range(D_in) if i not in static_indices]

    # La parte estática es constante. Añadimos dimensión de batch.
    x_static_data = torch.tensor(x_past[:, static_indices], dtype=torch.float32).unsqueeze(0).to(device)

    # La parte dinámica y su máscara evolucionarán en el bucle. Añadimos dimensión de batch.
    x_dynamic_seq = torch.tensor(x_past[:, dynamic_indices], dtype=torch.float32).unsqueeze(0).to(device)
    mask_dynamic_seq = torch.tensor(mask_past[:, dynamic_indices], dtype=torch.float32).unsqueeze(0).to(device)
    
    diag_init_t = torch.tensor(diag_init, dtype=torch.float32).view(1).to(device)
    outputs = []

    with torch.no_grad():
        for _ in range(n_future):
            # 2. Reconstruir la secuencia de entrada completa para el modelo en cada paso
            full_x_seq = torch.zeros(1, T, D_in, device=device, dtype=torch.float32)
            full_x_seq[:, :, dynamic_indices] = x_dynamic_seq
            full_x_seq[:, :, static_indices] = x_static_data

            # Reconstruir la máscara completa (la máscara para datos estáticos es siempre 1)
            # full_mask_seq = torch.zeros(1, T, D_in, device=device, dtype=torch.float32)
            full_mask_seq = torch.zeros(1, T, D_in_dyn, device=device, dtype=torch.float32)
            full_mask_seq[:, :, dynamic_indices] = mask_dynamic_seq
            # full_mask_seq[:, :, static_indices] = 1.0

            # 3. Llamar al modelo con la entrada y máscara completas (ambas 3D)
            # El modelo ahora recibe la máscara de la secuencia completa, no solo la dinámica
            pred_meas, pred_dx, _, _ = model(full_x_seq, full_mask_seq, diag_init_t)
            
            outputs.append(torch.cat([pred_meas, pred_dx], dim=1).cpu().numpy())

            # 4. Preparar la siguiente iteración actualizando SOLO la parte dinámica
            next_dynamic_slice = pred_meas.unsqueeze(1)
            next_mask_slice = torch.ones_like(next_dynamic_slice) # La predicción es un dato completo

            # Actualizar la secuencia y máscara dinámicas con una ventana deslizante
            x_dynamic_seq = torch.cat([x_dynamic_seq[:, 1:, :], next_dynamic_slice], dim=1)
            mask_dynamic_seq = torch.cat([mask_dynamic_seq[:, 1:, :], next_mask_slice], dim=1)

    y_pred_future = np.concatenate(outputs, axis=0)

    # Desescalar la parte de las medidas si se proporciona un scaler
    if scaler:
        num_meas = len(dynamic_indices)
        y_pred_future[:, :num_meas] = inverse_neuro_scaling(y_pred_future[:, :num_meas], scaler)

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



