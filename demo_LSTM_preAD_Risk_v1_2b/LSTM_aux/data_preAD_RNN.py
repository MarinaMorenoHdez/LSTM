# # -*- coding: utf-8 -*-
# """
# Created on Mon May 13 12:13:22 2024

# @author: cplatero
# """

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LSTM_aux.misce_LSTM import  preparar_secuencias, preparar_secuencias2

def load_data_RNN_bl(feat_names,min_visits,interv_months):
    ############################################################################
    # data
    ############################################################################
    if interv_months == 12:
        df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25.xlsx")
        df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_25.xlsx")  
    else:    
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_25_6m.xlsx")    
        # df_ = pd.read_excel("./data/preAD_LACT_ff_25_6m.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_25_6m.xlsx")  
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_ff_25_6m.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_ff_25_6m_LMCI.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_ff_25_6m_LMCI.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m_lineal.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_lineal_25_6m.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m_lineal_LMCI.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_lineal_25_6m_LMCI.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m_lineal_risk.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_lineal_25_6m_risk.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_ff_25_6m_Risk.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_ff_25_6m_Risk.xlsx")
        df_ = pd.read_excel("./data/preAD_NMs_ff_25_6m.xlsx")
        df2_ = pd.read_excel("./data/preAD_NMs_conNaN_excBsl_ff_25_6m.xlsx")

    
    # include age
    df_['Time'] = df_['AGE'] + df_['Year']
    df2_['Time'] = df2_['AGE'] + df2_['Year']
    
    df_['AGE'] = df_['Time']
    df2_['AGE'] = df2_['Time']
    
    # Contar el número de visitas por sujeto (RID)
    visitas_por_sujeto = df_['RID'].value_counts()
    
    # Fltrar los sujetos que tienen al menos tres visitas
    #min_visits = 3
    sujetos_mas_visitas = visitas_por_sujeto[visitas_por_sujeto >= min_visits].index.tolist()
    
    # Quedarse solo con los datos de los sujetos que tienen al menos tres visitas
    df = df_[df_['RID'].isin(sujetos_mas_visitas)]
    df2 = df2_[df2_['RID'].isin(sujetos_mas_visitas)]
    
    # Definir la columna RID como el identificador de los sujetos
    df = df.set_index('RID')
    df2 = df2.set_index('RID')
    
    # feat_names = ["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13"]
    # Realizar interpolación para llenar los valores NaN en los marcadores
    # imp = SimpleImputer(strategy='linear', missing_values=np.nan)
    # df[["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13"]] = imp.fit_transform(df[["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13"]])
    # df[feat_names] = df[feat_names].interpolate(method='linear', axis=0)
    
    # Obtener una lista de todos los RID únicos (sujetos)
    subjects = df.index.unique()
    
    # Dividir los sujetos en conjuntos de entrenamiento y prueba
    subjects_train, subjects_test = train_test_split(subjects, test_size=0.2, random_state=42)
    subjects_train_net, subjects_val = train_test_split(subjects_train, test_size=0.2, random_state=42)
    
    # Filtrar los datos para obtener los datos de entrenamiento y prueba
    df_train = df.loc[subjects_train_net]
    df_test = df.loc[subjects_test]
    df_val = df.loc[subjects_val]
    
    df2_train = df2.loc[subjects_train_net]
    df2_test = df2.loc[subjects_test]
    df2_val = df2.loc[subjects_val]
    
    # df_train = df_train.sort_values(by=['RID', 'M'], ascending=[True, True])
    # df_test = df_test.sort_values(by=['RID', 'M'], ascending=[True, True])
    # df_val = df_val.sort_values(by=['RID', 'M'], ascending=[True, True])
    
    df_train = df_train.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_test = df_test.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_val = df_val.sort_values(by=['RID', 'Year'], ascending=[True, True])
    
    df2_train = df2_train.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df2_test = df2_test.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df2_val = df2_val.sort_values(by=['RID', 'Year'], ascending=[True, True])
    
    
    
    features_train = df_train[feat_names]
    features_test = df_test[feat_names]
    features_val = df_val[feat_names]
    
    features2_train = df2_train[feat_names]
    features2_test = df2_test[feat_names]
    features2_val = df2_val[feat_names]
    
    # diagnosis_train = df_train[["DX"]]
    # diagnosis_test = df_test[["DX"]]
    # diagnosis_val = df_val[["DX"]]
    
    diagnosis_train = df_train[["DX_bl","Diag"]]
    diagnosis_test = df_test[["DX_bl","Diag"]]
    diagnosis_val = df_val[["DX_bl","Diag"]]
    
    diagnosis2_train = df2_train[["DX_bl","Diag"]]
    diagnosis2_test = df2_test[["DX_bl","Diag"]]
    diagnosis2_val = df2_val[["DX_bl","Diag"]]
    
    # # Crear un diccionario de mapeo para asignar valores numéricos a las categorías
    # mapeo = {'CN': 0, 'MCI': 1}
    # # Aplicar el mapeo a la columna categórica
    # diagnosis_train['Diag'] = diagnosis_train['DX'].apply(lambda x: mapeo[x])
    # diagnosis_test['Diag'] = diagnosis_test['DX'].apply(lambda x: mapeo[x])
    # diagnosis_val['Diag'] = diagnosis_val['DX'].apply(lambda x: mapeo[x])
    
    ############################################################################
    # Modelo
    ############################################################################
    # Normalizar los datos de entrenamiento
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(features_train)
    
    # Aplicar la misma normalización a los datos de test
    X_test_norm = scaler.transform(features_test)
    X_val_norm = scaler.transform(features_val)
    
    X2_test_norm = scaler.transform(features2_test)
    X2_train_norm = scaler.transform(features2_train)
    X2_val_norm = scaler.transform(features2_val)
    # # Calcular la media por columna, ignorando los NaN
    # media_por_columna = np.nanmean(X_test_norm, axis=0)
    # desviacion_estandar_por_columna = np.nanstd(X_test_norm, axis=0)
    
    # # Convertir los arrays normalizados de vuelta a DataFrames
    X_train_norm_df = pd.DataFrame(X_train_norm, index=features_train.index, columns=features_train.columns)
    X_test_norm_df = pd.DataFrame(X_test_norm, index=features_test.index, columns=features_test.columns)
    X_val_norm_df = pd.DataFrame(X_val_norm, index=features_val.index, columns=features_val.columns)
    
    X2_train_norm_df = pd.DataFrame(X2_train_norm, index=features2_train.index, columns=features2_train.columns)
    X2_test_norm_df = pd.DataFrame(X2_test_norm, index=features2_test.index, columns=features2_test.columns)
    X2_val_norm_df = pd.DataFrame(X2_val_norm, index=features2_val.index, columns=features2_val.columns)
    
    # # Para volver a la escala original desde la escala normalizada en los datos de entrenamiento
    # X_train_original = scaler.inverse_transform(X_train_norm_df)
    # X_test_original = scaler.inverse_transform(X_test_norm_df)
    
    
    # Combinar datos de marcadores y diagnóstico
    combined_train = pd.concat([X_train_norm_df, diagnosis_train[["DX_bl","Diag"]]], axis=1)
    combined_test = pd.concat([X_test_norm_df, diagnosis_test[["DX_bl","Diag"]]], axis=1)
    combined_val = pd.concat([X_val_norm_df, diagnosis_val[["DX_bl","Diag"]]], axis=1)
    
    combined2_train = pd.concat([X2_train_norm_df, diagnosis2_train[["DX_bl","Diag"]]], axis=1)
    combined2_test = pd.concat([X2_test_norm_df, diagnosis2_test[["DX_bl","Diag"]]], axis=1)
    combined2_val = pd.concat([X2_val_norm_df, diagnosis2_val[["DX_bl","Diag"]]], axis=1)
    
    # def preparar_secuencias(datos, longitud_secuencia):
    feat_names.append("DX_bl");       
    feat_names.append("Diag");    
    sec_train,ID_train = preparar_secuencias2(combined_train, min_visits,feat_names)
    sec_test,ID_test   = preparar_secuencias2(combined_test, min_visits,feat_names)
    sec_val, ID_val    = preparar_secuencias2(combined_val, min_visits,feat_names)
    
    sec2_train = preparar_secuencias(combined2_train, min_visits,feat_names)
    sec2_test = preparar_secuencias(combined2_test, min_visits,feat_names)
    sec2_val = preparar_secuencias(combined2_val, min_visits,feat_names)
    
    
    X_train, y_train = sec_train[:, :-1], sec_train[:, -1]
    X_test, y_test = sec_test[:, :-1], sec_test[:, -1]
    X_val, y_val = sec_val[:, :-1], sec_val[:, -1]
    
    
    X_train2 = sec2_train[:, :-1]
    X_test2 = sec2_test[:, :-1]
    X_val2 = sec2_val[:, :-1]
 
    
    # y2_train = sec2_train[:, -1]
    # y2_test = sec2_test[:, -1]
    
    mask_y_train_nonan =  ~np.isnan(sec2_train[:, -1])
    mask_y_test_nonan =  ~np.isnan(sec2_test[:, -1])
    mask_y_val_nonan =  ~np.isnan(sec2_val[:, -1])
    
    # Reestructurar los datos para que tengan la forma [muestras, pasos de tiempo, características]
    n_features=len(feat_names)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], n_features))
    
    X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], n_features))
    X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], n_features))
    X_val2 = np.reshape(X_val2, (X_val2.shape[0], X_val2.shape[1], n_features))
    
    mask_X_train_nonan =  ~np.isnan(X_train2)
    mask_X_test_nonan =  ~np.isnan(X_test2)
    mask_X_val_nonan =  ~np.isnan(X_val2)
    
    return X_train, X_test , X_val, y_train, y_test, y_val, mask_y_train_nonan, \
        mask_y_test_nonan, mask_y_val_nonan, mask_X_train_nonan, mask_X_test_nonan, mask_X_val_nonan, \
        ID_train, ID_test, ID_val, scaler




def load_data_RNN(feat_names,min_visits,interv_months):
    ############################################################################
    # data
    ############################################################################
    if interv_months == 12:
        df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25.xlsx")
        df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_25.xlsx")  
    else:    
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_25_6m.xlsx")    
        # df_ = pd.read_excel("./data/preAD_LACT_ff_25_6m.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_25_6m.xlsx")  
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_ff_25_6m.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_ff_25_6m_LMCI.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_ff_25_6m_LMCI.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m_lineal.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_lineal_25_6m.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m_lineal_LMCI.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_lineal_25_6m_LMCI.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_sinNaN_25_6m_lineal_risk.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_lineal_25_6m_risk.xlsx")
        # df_ = pd.read_excel("./data/preAD_LACT_ff_25_6m_Risk.xlsx")
        # df2_ = pd.read_excel("./data/preAD_LACT_conNaN_excBsl_ff_25_6m_Risk.xlsx")
        df_ = pd.read_excel("./data/preAD_NMs_ff_25_6m.xlsx")
        df2_ = pd.read_excel("./data/preAD_NMs_conNaN_excBsl_ff_25_6m.xlsx")

    
    # include age
    df_['Time'] = df_['AGE'] + df_['Year']
    df2_['Time'] = df2_['AGE'] + df2_['Year']
    
    df_['AGE'] = df_['Time']
    df2_['AGE'] = df2_['Time']
    
    # Contar el número de visitas por sujeto (RID)
    visitas_por_sujeto = df_['RID'].value_counts()
    
    # Fltrar los sujetos que tienen al menos tres visitas
    #min_visits = 3
    sujetos_mas_visitas = visitas_por_sujeto[visitas_por_sujeto >= min_visits].index.tolist()
    
    # Quedarse solo con los datos de los sujetos que tienen al menos tres visitas
    df = df_[df_['RID'].isin(sujetos_mas_visitas)]
    df2 = df2_[df2_['RID'].isin(sujetos_mas_visitas)]
    
    # Definir la columna RID como el identificador de los sujetos
    df = df.set_index('RID')
    df2 = df2.set_index('RID')
    
    # feat_names = ["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13"]
    # Realizar interpolación para llenar los valores NaN en los marcadores
    # imp = SimpleImputer(strategy='linear', missing_values=np.nan)
    # df[["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13"]] = imp.fit_transform(df[["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13"]])
    # df[feat_names] = df[feat_names].interpolate(method='linear', axis=0)
    
    # Obtener una lista de todos los RID únicos (sujetos)
    subjects = df.index.unique()
    
    # Dividir los sujetos en conjuntos de entrenamiento y prueba
    subjects_train, subjects_test = train_test_split(subjects, test_size=0.2, random_state=42)
    subjects_train_net, subjects_val = train_test_split(subjects_train, test_size=0.2, random_state=42)
    
    # Filtrar los datos para obtener los datos de entrenamiento y prueba
    df_train = df.loc[subjects_train_net]
    df_test = df.loc[subjects_test]
    df_val = df.loc[subjects_val]
    
    df2_train = df2.loc[subjects_train_net]
    df2_test = df2.loc[subjects_test]
    df2_val = df2.loc[subjects_val]
    
    # df_train = df_train.sort_values(by=['RID', 'M'], ascending=[True, True])
    # df_test = df_test.sort_values(by=['RID', 'M'], ascending=[True, True])
    # df_val = df_val.sort_values(by=['RID', 'M'], ascending=[True, True])
    
    df_train = df_train.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_test = df_test.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_val = df_val.sort_values(by=['RID', 'Year'], ascending=[True, True])
    
    df2_train = df2_train.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df2_test = df2_test.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df2_val = df2_val.sort_values(by=['RID', 'Year'], ascending=[True, True])
    
    
    
    features_train = df_train[feat_names]
    features_test = df_test[feat_names]
    features_val = df_val[feat_names]
    
    features2_train = df2_train[feat_names]
    features2_test = df2_test[feat_names]
    features2_val = df2_val[feat_names]
    
    # diagnosis_train = df_train[["DX"]]
    # diagnosis_test = df_test[["DX"]]
    # diagnosis_val = df_val[["DX"]]
    
    diagnosis_train = df_train[["Diag"]]
    diagnosis_test = df_test[["Diag"]]
    diagnosis_val = df_val[["Diag"]]
    
    diagnosis2_train = df2_train[["Diag"]]
    diagnosis2_test = df2_test[["Diag"]]
    diagnosis2_val = df2_val[["Diag"]]
    
    # # Crear un diccionario de mapeo para asignar valores numéricos a las categorías
    # mapeo = {'CN': 0, 'MCI': 1}
    # # Aplicar el mapeo a la columna categórica
    # diagnosis_train['Diag'] = diagnosis_train['DX'].apply(lambda x: mapeo[x])
    # diagnosis_test['Diag'] = diagnosis_test['DX'].apply(lambda x: mapeo[x])
    # diagnosis_val['Diag'] = diagnosis_val['DX'].apply(lambda x: mapeo[x])
    
    ############################################################################
    # Modelo
    ############################################################################
    # Normalizar los datos de entrenamiento
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(features_train)
    
    # Aplicar la misma normalización a los datos de test
    X_test_norm = scaler.transform(features_test)
    X_val_norm = scaler.transform(features_val)
    
    X2_test_norm = scaler.transform(features2_test)
    X2_train_norm = scaler.transform(features2_train)
    X2_val_norm = scaler.transform(features2_val)
    # # Calcular la media por columna, ignorando los NaN
    # media_por_columna = np.nanmean(X_test_norm, axis=0)
    # desviacion_estandar_por_columna = np.nanstd(X_test_norm, axis=0)
    
    # # Convertir los arrays normalizados de vuelta a DataFrames
    X_train_norm_df = pd.DataFrame(X_train_norm, index=features_train.index, columns=features_train.columns)
    X_test_norm_df = pd.DataFrame(X_test_norm, index=features_test.index, columns=features_test.columns)
    X_val_norm_df = pd.DataFrame(X_val_norm, index=features_val.index, columns=features_val.columns)
    
    X2_train_norm_df = pd.DataFrame(X2_train_norm, index=features2_train.index, columns=features2_train.columns)
    X2_test_norm_df = pd.DataFrame(X2_test_norm, index=features2_test.index, columns=features2_test.columns)
    X2_val_norm_df = pd.DataFrame(X2_val_norm, index=features2_val.index, columns=features2_val.columns)
    
    # # Para volver a la escala original desde la escala normalizada en los datos de entrenamiento
    # X_train_original = scaler.inverse_transform(X_train_norm_df)
    # X_test_original = scaler.inverse_transform(X_test_norm_df)
    
    
    # Combinar datos de marcadores y diagnóstico
    combined_train = pd.concat([X_train_norm_df, diagnosis_train[["Diag"]]], axis=1)
    combined_test = pd.concat([X_test_norm_df, diagnosis_test[["Diag"]]], axis=1)
    combined_val = pd.concat([X_val_norm_df, diagnosis_val[["Diag"]]], axis=1)
    
    combined2_train = pd.concat([X2_train_norm_df, diagnosis2_train[["Diag"]]], axis=1)
    combined2_test = pd.concat([X2_test_norm_df, diagnosis2_test[["Diag"]]], axis=1)
    combined2_val = pd.concat([X2_val_norm_df, diagnosis2_val[["Diag"]]], axis=1)
    
    # def preparar_secuencias(datos, longitud_secuencia):
    feat_names.append("Diag");    
    sec_train,ID_train = preparar_secuencias2(combined_train, min_visits,feat_names)
    sec_test,ID_test   = preparar_secuencias2(combined_test, min_visits,feat_names)
    sec_val, ID_val    = preparar_secuencias2(combined_val, min_visits,feat_names)
    
    sec2_train = preparar_secuencias(combined2_train, min_visits,feat_names)
    sec2_test = preparar_secuencias(combined2_test, min_visits,feat_names)
    sec2_val = preparar_secuencias(combined2_val, min_visits,feat_names)
    
    
    X_train, y_train = sec_train[:, :-1], sec_train[:, -1]
    X_test, y_test = sec_test[:, :-1], sec_test[:, -1]
    X_val, y_val = sec_val[:, :-1], sec_val[:, -1]
    
    
    X_train2 = sec2_train[:, :-1]
    X_test2 = sec2_test[:, :-1]
    X_val2 = sec2_val[:, :-1]
 
    
    # y2_train = sec2_train[:, -1]
    # y2_test = sec2_test[:, -1]
    
    mask_y_train_nonan =  ~np.isnan(sec2_train[:, -1])
    mask_y_test_nonan =  ~np.isnan(sec2_test[:, -1])
    mask_y_val_nonan =  ~np.isnan(sec2_val[:, -1])
    
    # Reestructurar los datos para que tengan la forma [muestras, pasos de tiempo, características]
    n_features=len(feat_names)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], n_features))
    
    X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], n_features))
    X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], n_features))
    X_val2 = np.reshape(X_val2, (X_val2.shape[0], X_val2.shape[1], n_features))
    
    mask_X_train_nonan =  ~np.isnan(X_train2)
    mask_X_test_nonan =  ~np.isnan(X_test2)
    mask_X_val_nonan =  ~np.isnan(X_val2)
    
    return X_train, X_test , X_val, y_train, y_test, y_val, mask_y_train_nonan, \
        mask_y_test_nonan, mask_y_val_nonan, mask_X_train_nonan, mask_X_test_nonan, mask_X_val_nonan, \
        ID_train, ID_test, ID_val, scaler




def load_data_NN(feat_names):
    #df = pd.read_excel("./data/preAD_LACT_EA_sinNaN_1y.xlsx")
    df = pd.read_excel("./data/preAD_LACT_EA_interp.xlsx")
    
    # include age
    df['Time'] = df['AGE'] + df['Year']
    
    
    # Definir la columna RID como el identificador de los sujetos
    df = df.set_index('RID')
    
    # Obtener una lista de todos los RID únicos (sujetos)
    subjects = df.index.unique()
    
    # Dividir los sujetos en conjuntos de entrenamiento y prueba
    subjects_train, subjects_test = train_test_split(subjects, test_size=0.2, random_state=42)
    subjects_train_net, subjects_val = train_test_split(subjects_train, test_size=0.2, random_state=42)
    
    # Filtrar los datos para obtener los datos de entrenamiento y prueba
    df_train = df.loc[subjects_train_net]
    df_test = df.loc[subjects_test]
    df_val = df.loc[subjects_val]
    
    
    df_train = df_train.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_test = df_test.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_val = df_val.sort_values(by=['RID', 'Year'], ascending=[True, True])
    
    feat_names_diag=feat_names.copy()
    feat_names_diag.append('Diag')
    
    df_train=df_train[feat_names_diag]
    df_test=df_test[feat_names_diag]
    df_val=df_val[feat_names_diag]
    
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    df_val = df_val.dropna()
    
    
    features_train = df_train[feat_names]
    features_test = df_test[feat_names]
    features_val = df_val[feat_names]
    
    
    # Normalizar los datos de entrenamiento
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train)
    
    # Aplicar la misma normalización a los datos de test
    X_test = scaler.transform(features_test)
    X_val = scaler.transform(features_val)
    
    
    y_train = df_train[["Diag"]].values
    y_test = df_test[["Diag"]].values
    y_val = df_val[["Diag"]].values

    return X_train, X_test , X_val, y_train, y_test, y_val, scaler



def load_data_df(feat_names):
    df = pd.read_excel("./data/preAD_LACT_EA_interp.xlsx")
    
    # include age
    df['Time'] = df['AGE'] + df['Year']
    
    
    # Definir la columna RID como el identificador de los sujetos
    df = df.set_index('RID')
    
    # Obtener una lista de todos los RID únicos (sujetos)
    subjects = df.index.unique()
    
    # Dividir los sujetos en conjuntos de entrenamiento y prueba
    subjects_train, subjects_test = train_test_split(subjects, test_size=0.2, random_state=42)
    subjects_train_net, subjects_val = train_test_split(subjects_train, test_size=0.2, random_state=42)
    
    # Filtrar los datos para obtener los datos de entrenamiento y prueba
    df_train = df.loc[subjects_train_net]
    df_test = df.loc[subjects_test]
    df_val = df.loc[subjects_val]
    
    
    df_train = df_train.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_test = df_test.sort_values(by=['RID', 'Year'], ascending=[True, True])
    df_val = df_val.sort_values(by=['RID', 'Year'], ascending=[True, True])
    
    feat_names_diag=feat_names.copy()
    feat_names_diag.append('Diag')
    feat_names_diag.append('Time')
    
    df_train=df_train[feat_names_diag]
    df_test=df_test[feat_names_diag]
    df_val=df_val[feat_names_diag]
    
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    df_val = df_val.dropna()
    
    
    features_train = df_train[feat_names]
    features_test = df_test[feat_names]
    features_val = df_val[feat_names]
    
    
    # Normalizar los datos de entrenamiento
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train)
    
    # Aplicar la misma normalización a los datos de test
    X_test = scaler.transform(features_test)
    X_val = scaler.transform(features_val)
    
    df_train_nor = df_train.copy()
    df_test_nor = df_test.copy()
    df_val_nor = df_val.copy()
    
    df_train_nor[feat_names]=X_train
    df_test_nor[feat_names]=X_test
    df_val_nor[feat_names]=X_val
    
    #X_val_orig=scaler.inverse_transform(X_val)
       

    return df_train, df_test , df_val, df_train_nor, df_test_nor , df_val_nor, scaler

