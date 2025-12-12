import os
new_directory = "C:/Users/cplatero/OneDrive - Universidad Polit�cnica de Madrid/src/DeepL/LSTM/v1_2_risk/demo"
# new_directory = "C:/Users/User/OneDrive - Universidad Polit�cnica de Madrid/src/DeepL/LSTM/v1_2_risk/demo"
os.chdir(new_directory)
import numpy as np
import torch
import pickle


algorit = 3


############################################################################
# data
############################################################################
from LSTM_aux.data_preAD_RNN import  load_data_RNN, load_data_RNN_bl

   

if algorit <= 6:
    feat_names = ["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13", "Time", "PTEDUCAT", "APOE4", "Sex"]  
    static_indices = [4, 5, 6, 7]
    static_indices_plus = [4, 5, 6, 7, 8]
    monotonic_signs = [-1, +1, +1, +1]
    
    
num_feat=len(feat_names)

## new sequence data
# min_visits = 4
# interv_months = 6 
# X_train, X_test , X_val, y_train, y_test, y_val, mask_y_train_nonan, \
#     mask_y_test_nonan,  mask_y_val_nonan, mask_X_train_nonan, mask_X_test_nonan, mask_X_val_nonan, \
#     ID_train , ID_test , ID_val , scaler = load_data_RNN_bl(feat_names,min_visits,interv_months)
    
# data = {"X_train" : X_train, 
#         "X_val" : X_val,
#         "X_test" : X_test,
#         "y_train" : y_train, 
#         "y_val" : y_val,
#         "y_test" : y_test,
#         "mask_y_train_nonan" : mask_y_train_nonan,
#         "mask_y_val_nonan" : mask_y_val_nonan,
#         "mask_y_test_nonan" : mask_y_test_nonan,
#         "mask_X_train_nonan" : mask_X_train_nonan,
#         "mask_X_val_nonan" : mask_X_val_nonan,
#         "mask_X_test_nonan" : mask_X_test_nonan,
#         "ID_train" : ID_train,
#         "ID_val" : ID_val,
#         "ID_test" : ID_test,
#         "scaler" : scaler}
            
# with open("./data/data_LACT_risk.pkl", "wb") as file:
#     pickle.dump(data, file)
 
with open("./data/data_LACT_risk.pkl", "rb") as file:
    data = pickle.load(file)






############################################################################
# sequence
############################################################################
     
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



############################################################################
# Models 
############################################################################

if algorit ==1:
    from LSTM_aux.RiskAugmentedLSTM_deterministic import RiskAugmentedLSTM_deterministic 
    from LSTM_aux.train_eval_interp_risk_smooth import train_risk_augmented_model_smooth, evaluate_risk_augmented_model
    name_model = "./output/LSTM_determ_res_mono_risk_interp_smooth_0.pt"    


elif algorit ==2:
    from LSTM_aux.ImputationEDCFAugmentedLSTM import ImputationAugmentedLSTM
    from LSTM_aux.train_eval_EDCF_risk_smooth import train_imputation_augmented_model, evaluate_imputation_augmented_model
    name_model = "./output/LSTM_determ_res_mono_risk_edcf_optuna.pt"
    
elif algorit ==3:
    from LSTM_aux.Risk_LSTM_TSGAIN_determ import RiskAugmentedLSTM_TSGAIN_deterministic
    from LSTM_aux.train_eval_TSGAIN_risk_smooth import train_tsgain_model_smooth, evaluate_tsgain_model
    name_model = "./output/LSTM_determ_res_mono_risk_TSGAIN_smooth.pt"
    
    
    
# Entrenamiento
diag_init_train = X_train[:, 0, -1]
X_train = X_train[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido
diag_init_val = X_val[:, 0, -1]
X_val = X_val[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido
diag_init_test = X_test[:, 0, -1]
X_test = X_test[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido
diag_init_all = X_all[:, 0, -1]
X_all = X_all[:, :, :-1]  # quitar diagnóstico de la entrada si está incluido


# Parámetros del modelo
input_size = X_train.shape[2]
hidden_size = 64
dropout_rate = 0.3
num_layers = 2
seed = 42

lr=1e-3   



# New model                            
if algorit ==1:
    model = RiskAugmentedLSTM_deterministic(input_size=input_size,
                                            static_indices=static_indices,
                                            hidden_size=hidden_size,
                                            dropout_rate=dropout_rate,
                                            num_layers=num_layers,
                                            seed=seed)
 
elif algorit == 2:
    model = ImputationAugmentedLSTM(
        input_size=input_size,
        static_indices=static_indices,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_layers=num_layers
    )
    

elif algorit == 3:
    model = RiskAugmentedLSTM_TSGAIN_deterministic(
        input_size=input_size,
        static_indices=static_indices,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        num_layers=num_layers
    )


# train & eval                             
if algorit >1:
    num_dyn=input_size-len(static_indices)
    mask_X_train_nonan=mask_X_train_nonan[:, :, :num_dyn]
    mask_X_val_nonan=mask_X_val_nonan[:, :, :num_dyn]
    mask_X_test_nonan=mask_X_test_nonan[:, :, :num_dyn]
    mask_X_all=mask_X_all[:, :, :num_dyn]
 
 

if algorit ==1:
    lambda_clf=1
    lambda_mono=0.1
    lambda_smooth=0.1
    

    model = train_risk_augmented_model_smooth(
        model,
        X_train, y_train, mask_y_train_nonan, ID_train[:,0], diag_init_train,
        X_val, y_val, mask_y_val_nonan, ID_val[:,0], diag_init_val,
        num_epochs=1000,
        lr=lr,
        patience=20,
        lambda_clf=lambda_clf,
        lambda_mono=lambda_mono,
        lambda_smooth=lambda_smooth,
        monotonic_signs= monotonic_signs  
    )
    
    # Evaluaci�n
    evaluate_risk_augmented_model(model, X_test, y_test, mask_y_test_nonan, ID_test, diag_init_test, scaler) 
    evaluate_risk_augmented_model(model, X_all, y_all, mask_y_all, ID_all, diag_init_all, scaler) 

    


elif algorit ==2:
    # lambda_clf: 0.05 — 0.2 (0.5 suele ser alto; reduce si quieres mejores medidas).
    # lambda_impute: 0.5 — 2.0 (2 puede ser alto si tu EDCF ya hace MSE en input scale).
    # lambda_mono: 0.01 — 0.2 (1.0 es grande; empieza pequeño y sube).
    # lambda_smooth: 0.01 — 0.1 (0.5 hará trajectorias muy planas).
    # Empieza con lambda_clf=0.1, lambda_impute=1.0, lambda_mono=0.05, lambda_smooth=0.05.
    lambda_clf =  0.2 #0.1 
    lambda_impute = 2.0 #1.0 
    lambda_mono = 0.5 #0.05 
    lambda_smooth = 0.05 #0.1 
    
    # lambda_clf = trial.suggest_categorical("lambda_clf", [0.1, 0.2, 0.5])
    # lambda_impute = trial.suggest_categorical("lambda_impute", [0.5, 1.0, 2.0, 5.0])
    # lambda_mono = trial.suggest_categorical("lambda_mono", [0.1, 0.5, 1.0, 2.0])
    # lambda_smooth = trial.suggest_categorical("lambda_smooth", [0.01, 0.05, 0.1, 0.5])
    
    # Hiperpar�metros �ptimos: {'lambda_clf': 0.1, 'lambda_impute': 5.0, 'lambda_mono': 0.1, 'lambda_smooth': 0.01}
    # Mejor valor de validaci�n: 0.6162495613098145

    # lambda_clf =  0.1 #0.1 
    # lambda_impute = 5.0 #1.0 
    # lambda_mono = 0.1 #0.05 
    # lambda_smooth = 0.01 #0.1 
    
    
    
    
    
    model = train_imputation_augmented_model(
        model,
        X_train, y_train, mask_y_train_nonan, mask_X_train_nonan, ID_train[:,0], diag_init_train,
        X_val, y_val, mask_y_val_nonan, mask_X_val_nonan, ID_val[:,0], diag_init_val,
        num_epochs=1000,
        lr=lr,
        patience=15,
        lambda_clf=lambda_clf,
        lambda_impute=lambda_impute, # Ajusta este hiperparámetro para balancear la pérdida de imputación
        lambda_mono=lambda_mono,
        lambda_smooth=lambda_smooth,
        monotonic_signs= monotonic_signs
    )
    evaluate_imputation_augmented_model(model, X_test, y_test, mask_y_test_nonan, mask_X_test_nonan, ID_test, diag_init_test, scaler)
    evaluate_imputation_augmented_model(model, X_all, y_all, mask_y_all, mask_X_all, ID_all, diag_init_all, scaler)
    

elif algorit ==3:
    lambda_clf =  0.5 # 0.1 [0.1, 0.2, 0.5]
    lambda_impute = 2 # 5 [0.5, 1.0, 2.0, 5.0]
    lambda_mono = 1.0 # [0.1, 0.5, 1.0, 2.0]
    lambda_smooth = 0.5 # [0.01, 0.05, 0.1, 0.5]

    model = train_tsgain_model_smooth(
        model,
        X_train, y_train, mask_y_train_nonan, mask_X_train_nonan, ID_train[:,0], diag_init_train,
        X_val, y_val, mask_y_val_nonan, mask_X_val_nonan, ID_val[:,0], diag_init_val,
        num_epochs=1000,
        lr=lr,
        patience=15,
        lambda_impute=lambda_impute,
        lambda_clf=lambda_clf,
        lambda_mono=lambda_mono,
        lambda_smooth=lambda_smooth,
        monotonic_signs= monotonic_signs  # Ajusta este hiperparámetro para balancear la pérdida de imputación
    )
    evaluate_tsgain_model(model, X_test, y_test, mask_y_test_nonan, mask_X_test_nonan, ID_test, diag_init_test, scaler)
    evaluate_tsgain_model(model, X_all, y_all, mask_y_all, mask_X_all, ID_all, diag_init_all, scaler)

    
    
torch.save(model.state_dict(), name_model)


    
