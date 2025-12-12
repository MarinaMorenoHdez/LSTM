import os
new_directory = "C:/Users/cplatero/OneDrive - Universidad Politécnica de Madrid/src/DeepL/LSTM/v1_2_risk/demo"
# new_directory = "C:/Users/User/OneDrive - Universidad Politécnica de Madrid/src/DeepL/LSTM/v1_2_risk"
os.chdir(new_directory)


import numpy as np
import torch
import pickle



############################################################################
# data
############################################################################

algorit = 3
subject_id = 113 # 2, 21, 113, 55, 56, 59 61, 66, 72
n_future=10


if algorit <= 6:
    feat_names = ["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13", "Time", "PTEDUCAT", "APOE4", "Sex"]  
    static_indices = [4, 5, 6, 7]
    static_indices_plus = [4, 5, 6, 7, 8]
    monotonic_signs = [-1, +1, +1, +1]
    


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
    from LSTM_aux.train_eval_interp_risk_smooth import evaluate_risk_augmented_model    
    name_model = "./output/LSTM_determ_res_mono_risk_interp_smooth_0.pt"
    

elif algorit ==2:
    from LSTM_aux.ImputationEDCFAugmentedLSTM import ImputationAugmentedLSTM
    from LSTM_aux.train_eval_EDCF_risk_smooth import  evaluate_imputation_augmented_model    
    name_model = "./output/LSTM_determ_res_mono_risk_edcf_optuna.pt"
     
elif algorit ==3:
    from LSTM_aux.Risk_LSTM_TSGAIN_determ import RiskAugmentedLSTM_TSGAIN_deterministic
    from LSTM_aux.train_eval_TSGAIN_risk_smooth import  evaluate_tsgain_model
    name_model = "./output/LSTM_determ_res_mono_risk_TSGAIN_smooth.pt"    
    
elif algorit == 4:
    # Importar el nuevo modelo
    from LSTM_aux.RGRU_EDCF import RGRU_EDCF
    # Puedes necesitar un nuevo script de evaluación o adaptar uno existente
    # from LSTM_aux.train_eval_Jia25 import train_jia25_model, evaluate_jia25
    name_model = "./output/RGRU_EDCF_model.pt"
    
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
# lambda_clf=1
# lambda_mono=0.1
# lambda_smooth=0.1
# lambda_impute=0.5



# Crear modelo
if algorit ==1:
    model = RiskAugmentedLSTM_deterministic(input_size=input_size,
                                            static_indices=static_indices,
                                            hidden_size=hidden_size,
                                            dropout_rate=dropout_rate,
                                            num_layers=num_layers,
                                            seed=seed)
    
   
elif algorit ==2:
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

elif algorit == 4:
    model = RGRU_EDCF(
        input_size=input_size,
        hidden_size=hidden_size, # Jia usa 64 por defecto
        num_layers=num_layers,
        dropout=dropout_rate
    )
                              
    
        
   
   
# load model
model.load_state_dict(torch.load(name_model, map_location=torch.device("cpu")))
model.eval()

# eval                             
if algorit >1:
    num_dyn=input_size-len(static_indices)
    mask_X_train_nonan=mask_X_train_nonan[:, :, :num_dyn]
    mask_X_val_nonan=mask_X_val_nonan[:, :, :num_dyn]
    mask_X_test_nonan=mask_X_test_nonan[:, :, :num_dyn]
    mask_X_all=mask_X_all[:, :, :num_dyn]
 

if algorit ==1:  
    evaluate_risk_augmented_model(model, X_test, y_test, mask_y_test_nonan, ID_test, diag_init_test, scaler) 
    evaluate_risk_augmented_model(model, X_all, y_all, mask_y_all, ID_all, diag_init_all, scaler)  
    
elif algorit ==5:
    evaluate_imputation_augmented_model(model, X_test, y_test, mask_y_test_nonan, mask_X_test_nonan, ID_test, diag_init_test, scaler)
    evaluate_imputation_augmented_model(model, X_all, y_all, mask_y_all, mask_X_all, ID_all, diag_init_all, scaler)

elif algorit ==6:    
    evaluate_tsgain_model(model, X_test, y_test, mask_y_test_nonan, mask_X_test_nonan, ID_test, diag_init_test, scaler)
    evaluate_tsgain_model(model, X_all, y_all, mask_y_all, mask_X_all, ID_all, diag_init_all, scaler)

elif algorit == 4:
    # Aquí necesitarás un bucle de entrenamiento que soporte la función de pérdida combinada
    # descrita en la Ec. 26 del paper (Regresión + Clasificación + Términos cruzados)
    pass

# Personalization
# Subject
mask_subj = (ID_all[:, 0] == subject_id) #ok


x_subj = X_all[mask_subj,:,:]
y_subj = y_all[mask_subj,:]
diag_init_subj=diag_init_all[mask_subj]
ID_subj=ID_all[mask_subj,:]
mask_y_subj = mask_y_all[mask_subj,-1]

if algorit > 1:
    mask_X_test = mask_X_all[mask_subj,:,:]
    mask_X_test=mask_X_test[:, :, :num_dyn]
    # mask_X_test[:,0,:] = 1       
   
if algorit ==1:
    from LSTM_aux.predictive_lstm_interp_inc_risk import predictive_lstm_interp_risk, plot_subject_trajectory
    y_true_neuro, y_pred_neuro, y_true, y_pred = predictive_lstm_interp_risk(model, x_subj, y_subj, diag_init_subj, scaler)
elif algorit ==2:
    from LSTM_aux.predictive_lstm_edcf_inc_risk import predictive_lstm_edcf_risk, plot_subject_trajectory
    y_true_neuro, y_pred_neuro, y_true, y_pred = predictive_lstm_edcf_risk(model, x_subj, mask_X_test, y_subj, diag_init_subj, scaler)
elif algorit ==3:  
    from LSTM_aux.predictive_lstm_TSGAIN_inc_risk import predictive_lstm_TSGAIN_risk, plot_subject_trajectory
    y_true_neuro, y_pred_neuro, y_true, y_pred = predictive_lstm_TSGAIN_risk(model, x_subj, mask_X_test, y_subj, diag_init_subj, scaler)
    

mask_subj = mask_y_subj
y_true_neuro_subj=y_true_neuro[mask_subj,:]
y_pred_neuro_subj=y_pred_neuro[mask_subj,:]
y_true_sub=y_true[mask_subj]
y_pred_sub=y_pred[mask_subj]
ID_subj=ID_subj[mask_subj,:]

y_true_neuro_subj= np.concatenate([y_true_neuro_subj,y_true_sub.reshape(-1, 1)], axis=1)
y_pred_neuro_subj= np.concatenate([y_pred_neuro_subj,y_pred_sub.reshape(-1, 1)], axis=1)

# plot_subject_trajectory(y_true_neuro_subj, y_pred_neuro_subj, feat_names, ID_subj[:,-1])


## Future

x_end = x_subj[-1,:,:]
diag_init_end=diag_init_subj[-1]

if algorit ==1:
    from LSTM_aux.predictive_lstm_interp_inc_risk import predict_full_trajectory_risk, plot_subject_trajectory_with_future, inverse_neuro_scaling
    y_pred_all = predict_full_trajectory_risk(model, x_end, diag_init_end, n_future, static_indices, device='cpu') 
    # from LSTM_aux.predictive_lstm_interp_inc import predict_full_trajectory, plot_subject_trajectory_with_future, inverse_neuro_scaling
    # y_pred_all = predict_full_trajectory(model, x_end, diag_init_end, n_future)                
elif algorit ==2:
    mask_X_end=mask_X_test[-1,:,:num_dyn]
    mask_X_end[0,:] = 1
    from LSTM_aux.predictive_lstm_edcf_inc_risk import predict_full_trajectory_risk, plot_subject_trajectory_with_future, inverse_neuro_scaling
    y_pred_all = predict_full_trajectory_risk(model, x_end, mask_X_end, diag_init_end, n_future, static_indices, device='cpu')
elif algorit ==3:
    mask_X_end=mask_X_test[-1,:,:num_dyn]
    #  mask_X_end[0,:] = 1
    from LSTM_aux.predictive_lstm_TSGAIN_inc_risk import predict_full_trajectory_risk, plot_subject_trajectory_with_future, inverse_neuro_scaling
    y_pred_all = predict_full_trajectory_risk(model, x_end, mask_X_end, diag_init_end, n_future, static_indices, device='cpu')
    
    
num_feat=len(feat_names)-len(static_indices)
y_pred_all_NMs=inverse_neuro_scaling(y_pred_all[:, 0:num_feat], scaler)

y_pred_all_orig = np.concatenate([y_pred_all_NMs, y_pred_all[:, -1].reshape(-1, 1)], axis=1)

visit_times_past = max(ID_subj[:,-1])                      # o tus tiempos reales si los tienes
visit_times_future = np.arange(visit_times_past, visit_times_past + n_future)      # futuras visitas
visit_times = np.concatenate([ID_subj[:,-1], visit_times_future])

plot_subject_trajectory_with_future(y_true_neuro_subj, y_pred_neuro_subj, y_pred_all_orig, feat_names, visit_times)
    
    

    
    


    
