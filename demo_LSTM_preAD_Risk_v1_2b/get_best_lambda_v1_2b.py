# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 18:15:43 2025

@author: cplatero
"""

import optuna

import os
new_directory = "C:/Users/cplatero/OneDrive - Universidad Politécnica de Madrid/src/DeepL/LSTM/v1_2_risk/demo"
os.chdir(new_directory)
import numpy as np
import torch
import pickle



############################################################################
# data
############################################################################
algorit = 1

if algorit <= 6:
    feat_names = ["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13", "Time", "PTEDUCAT", "APOE4", "Sex"]  
    static_indices = [4, 5, 6, 7]
    static_indices_plus = [4, 5, 6, 7, 8]
    monotonic_signs = [-1, +1, +1, +1]

 
with open("./data/data_LACT_risk.pkl", "rb") as file:
    data = pickle.load(file)
     
X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]
y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]
mask_y_train_nonan = data["mask_y_train_nonan"]
mask_y_val_nonan = data["mask_y_val_nonan"]
mask_y_test_nonan = data["mask_y_test_nonan"]
mask_X_train_nonan = data["mask_X_train_nonan"]
mask_X_val_nonan = data["mask_X_val_nonan"]
mask_X_test_nonan = data["mask_X_test_nonan"]
ID_train = data["ID_train"]
ID_val = data["ID_val"]
ID_test = data["ID_test"]
scaler = data["scaler"]



y_train = np.delete(y_train, static_indices_plus, axis=1)
y_val = np.delete(y_val, static_indices_plus, axis=1)
y_test = np.delete(y_test, static_indices_plus, axis=1)
mask_y_train_nonan = np.delete(mask_y_train_nonan, static_indices_plus, axis=1)
mask_y_val_nonan = np.delete(mask_y_val_nonan, static_indices_plus, axis=1)
mask_y_test_nonan = np.delete(mask_y_test_nonan, static_indices_plus, axis=1)


X_train=X_train[:, :, :-1]
X_val=X_val[:, :, :-1]
X_test=X_test[:, :, :-1]
mask_X_train_nonan=mask_X_train_nonan[:, :, :-1]
mask_X_val_nonan=mask_X_val_nonan[:, :, :-1]
mask_X_test_nonan=mask_X_test_nonan[:, :, :-1]


X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)
mask_X_all = np.concatenate([mask_X_train_nonan, mask_X_val_nonan], axis=0)
mask_y_all = np.concatenate([mask_y_train_nonan, mask_y_val_nonan], axis=0)
ID_all = np.concatenate([ID_train, ID_val], axis=0)

diag_init_train = X_train[:, 0, -1]
X_train = X_train[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido
diag_init_val = X_val[:, 0, -1]
X_val = X_val[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido
diag_init_test = X_test[:, 0, -1]
X_test = X_test[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido
diag_init_all = X_all[:, 0, -1]
X_all = X_all[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido


############################################################################
# Interpolation 
############################################################################
input_size = X_train.shape[2]
# train & eval                             
if algorit >1:
    num_dyn=input_size-len(static_indices)
    mask_X_train_nonan=mask_X_train_nonan[:, :, :num_dyn]
    mask_X_val_nonan=mask_X_val_nonan[:, :, :num_dyn]
    mask_X_test_nonan=mask_X_test_nonan[:, :, :num_dyn]
    mask_X_all=mask_X_all[:, :, :num_dyn]
 
    # mask_X_train_nonan[:,0,:] = 1
    # mask_X_val_nonan[:,0,:] = 1
    # mask_X_test_nonan[:,0,:] = 1
    # mask_X_all[:,0,:] = 1    

if algorit ==1:
    from LSTM_aux.RiskAugmentedLSTM_deterministic import RiskAugmentedLSTM_deterministic
    from LSTM_aux.train_eval_interp_risk_smooth import train_risk_augmented_model_smooth_logged
    

elif algorit ==2:
    from LSTM_aux.ImputationEDCFAugmentedLSTM import ImputationAugmentedLSTM
    from LSTM_aux.train_eval_EDCF_risk_smooth import train_imputation_augmented_model_logged
    
# Parametros del modelo
input_size = X_train.shape[2]
hidden_size = 64
dropout_rate = 0.3
num_layers = 2
seed = 42
lr=1e-3   
    


def objective(trial, seed=42):

    # lambda_clf =  0.2 #0.1 
    # lambda_impute = 2.0 #1.0 
    # lambda_mono = 0.5 #0.05 
    # lambda_smooth = 0.05 #0.1 
    
    lambda_clf = trial.suggest_categorical("lambda_clf", [0.1, 0.2, 0.5])
    lambda_impute = trial.suggest_categorical("lambda_impute", [0.5, 1.0, 2.0, 5.0])
    lambda_mono = trial.suggest_categorical("lambda_mono", [0.1, 0.5, 1.0, 2.0])
    lambda_smooth = trial.suggest_categorical("lambda_smooth", [0.01, 0.05, 0.1, 0.5])

    if algorit ==2:
        model = ImputationAugmentedLSTM(
            input_size=input_size,
            static_indices=static_indices,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )
    
        model, history = train_imputation_augmented_model_logged(
            model,
            X_train, y_train, mask_y_train_nonan, mask_X_train_nonan, ID_train[:,0], diag_init_train,
            X_val, y_val, mask_y_val_nonan, mask_X_val_nonan, ID_val[:,0], diag_init_val,
            num_epochs=1000,
            lr=lr,
            patience=15,
            lambda_clf=lambda_clf,
            lambda_impute=lambda_impute, # Ajusta este hiperparÃ¡metro para balancear la pÃ©rdida de imputaciÃ³n
            lambda_mono=lambda_mono,
            lambda_smooth=lambda_smooth,
            monotonic_signs=monotonic_signs
        )
        
    # Retornar el mejor val_loss
    return min(history["val"])




study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial), n_trials=50)

# print("Mejores hiperparámetros:", study.best_params)

#######################################################
# save results
#######################################################

import json

# Carpeta donde guardar resultados
results_file = "optuna_results.json"

# Guardar todos los trials y mejores parámetros
with open(results_file, "w") as f:
    json.dump({
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value
            }
            for t in study.trials
        ]
    }, f, indent=4)

print(f"Resultados guardados en {results_file}")
print("Mejores hiperparámetros encontrados:", study.best_params)
print("Mejor valor de validación:", study.best_value)

#######################################################
# read results
#######################################################

import json

results_file = "optuna_results.json"

with open(results_file, "r") as f:
    results = json.load(f)

print("Hiperparámetros óptimos:", results["best_params"])
print("Mejor valor de validación:", results["best_value"])
print("Total de trials realizados:", len(results["trials"]))

# Si quieres recorrer todos los trials:
for trial in results["trials"][:5]:  # muestra los primeros 5
    print(f"Trial {trial['number']} | params={trial['params']} | value={trial['value']}")
