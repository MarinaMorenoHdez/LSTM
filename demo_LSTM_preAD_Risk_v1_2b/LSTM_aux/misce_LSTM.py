# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr  4 17:40:34 2024

# @author: cplatero
# """

import pandas as pd
import numpy as np
import tensorflow as tf



def preparar_secuencias(datos, longitud_secuencia, features_names):
    secuencias = []
    for sujeto, grupo in datos.groupby('RID'):
        # visitas = grupo[["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13", "Diag"]].values
        visitas = grupo[features_names].values
        for i in range(len(visitas)):           
            if i + longitud_secuencia <= len(visitas):
                secuencia = visitas[i:i + longitud_secuencia]
                secuencias.append(secuencia)
    return np.array(secuencias)



def preparar_secuencias2(datos, longitud_secuencia, features_names):
    secuencias = []
    data_subjs =[]
    for sujeto, grupo in datos.groupby('RID'):
        # visitas = grupo[["RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13", "Diag"]].values
        visitas = grupo[features_names].values
        for i in range(len(visitas)):           
            if i + longitud_secuencia <= len(visitas):
                secuencia = visitas[i:i + longitud_secuencia]
                secuencias.append(secuencia)
                data_subjs.append([sujeto,i])
    return np.array(secuencias), np.array(data_subjs)


def predict_future_LSTM(model, data, n_steps, future_steps, features_names):
    #future_data = np.copy(data[-n_steps:].reshape(1, n_steps, data.shape[1]))
    secuencias_sujeto = preparar_secuencias(data, n_steps, features_names)
    X_sujeto, y_sujeto = secuencias_sujeto[:, :-1], secuencias_sujeto[:, -1]
    X_sujeto = np.reshape(X_sujeto, (X_sujeto.shape[0], X_sujeto.shape[1], 5))

    for i in range(future_steps):
        predicciones_sujeto = model.predict(X_sujeto)
        # Futuro
        nueva_visita=np.array([predicciones_sujeto[-1]]).reshape(-1)
        
        #columnas = ['RID', "RAVLT_learning", "TRABSCOR", "CDRSB", "ADAS13", "Diag"]
        nueva_visita_dict = {}
        nueva_visita_dict['RID'] = data.index.unique()[0]  # Suponiendo que el RID está en la primera posición del vector
        for i, columna in enumerate(features_names[:]):  # Empezamos desde 1 porque ya hemos añadido el RID
            nueva_visita_dict[columna] = nueva_visita[i]
        
        # Convierte el diccionario en un DataFrame
        nueva_visita_df = pd.DataFrame([nueva_visita_dict])
        nueva_visita_df = nueva_visita_df.set_index('RID')
        # Añade la nueva visita al DataFrame existente
        data = pd.concat([data, nueva_visita_df], ignore_index=False)
        secuencias_sujeto = preparar_secuencias(data, n_steps, features_names)
        X_sujeto, y_sujeto = secuencias_sujeto[:, :-1], secuencias_sujeto[:, -1]
        X_sujeto = np.reshape(X_sujeto, (X_sujeto.shape[0], X_sujeto.shape[1], 5))

                

    return predicciones_sujeto



def estimate_nan_RNN_length_sec_2(X, Y, modelo, ID):
    
    X_train_estimado = []
    y_train_estimado = []
    
    X_train = X.copy()
    y_train = Y.copy()

    # Iterar sobre cada secuencia          
    # for i in range(0,100):
    for i in range(X_train.shape[0]):      
        # Verificar si hay NaN en y_train
        if np.isnan(y_train[i]).any():
            # Predecir los valores NaN en y_train usando el modelo LSTM
            y_pred = modelo.predict(X_train[i].reshape(1, 2, 5))
            y_train[i][np.isnan(y_train[i])] = y_pred.flatten()[np.isnan(y_train[i])]
            if i < len(X_train) - 1:
                if ID[i]==ID[i+1]:
                    X_train[i+1] = np.vstack((X_train[i+1][0,:], y_train[i]))
                    if i < len(X_train) - 2:
                        if ID[i]==ID[i+2]:
                            X_train[i+2] = np.vstack((y_train[i],X_train[i+2][1,:]))

        # Agregar y_train actualizado
        X_train_estimado.append(X_train[i])
        y_train_estimado.append(y_train[i])


    X_train_estimado=np.array(X_train_estimado)
    y_train_estimado= np.array(y_train_estimado)
    
    return X_train_estimado, y_train_estimado

 
def custom_loss_MAE_Cross_NaN(y_true, y_pred):
    # Crear una máscara para identificar valores nan en y_true
    mask = tf.math.is_nan(y_true)
    mask = ~mask
    y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
    mask = tf.cast(mask, tf.float32)  
    # loss = tf.reduce_sum(mask * tf.square(y_true - y_pred)) / tf.reduce_sum(mask)    
    
    # Obtener las dimensiones de los tensores de entrada
    #num_quantitative_features = 4
    num_quantitative_features = y_true.shape[1] - 1
    #num_binary_features = 1
    mask_feat = mask[:, :num_quantitative_features]
    mask_bin = mask[:, num_quantitative_features:]

    # Dividir las predicciones y las etiquetas verdaderas en partes cuantitativas y binarias
    y_true_quantitative, y_true_binary = y_true[:, :num_quantitative_features], y_true[:, num_quantitative_features:]
    y_pred_quantitative, y_pred_binary = y_pred[:, :num_quantitative_features], y_pred[:, num_quantitative_features:]

    # # Calcular el error cuadrático medio para las características cuantitativas
    # mse_quantitative = K.mean(K.square(y_true_quantitative - y_pred_quantitative), axis=-1)

    # # Calcular la entropía cruzada binaria para la característica binaria
    # binary_crossentropy = K.mean(K.binary_crossentropy(y_true_binary, y_pred_binary), axis=-1)

    # # Ponderar las pérdidas y combinarlas
    # weighted_loss = mse_quantitative * 0.8 + binary_crossentropy * 0.2
    # Calcular la pérdida de las variables cuantitativas (MAE ponderado)
    #mae_loss = tf.reduce_mean(tf.abs(y_pred_quantitative - y_true_quantitative), axis=-1)
    mae_loss = tf.reduce_sum(  mask_feat * tf.abs(y_pred_quantitative - y_true_quantitative),  axis=-1) / tf.reduce_sum(mask_feat)    
    
    weighted_mae_loss = tf.reduce_sum(mae_loss, axis=-1)  # Sumar las pérdidas de las cuatro variables
    
    # Calcular la pérdida de la variable binaria (entropía cruzada binaria)
    # binary_loss = tf.keras.losses.binary_crossentropy(y_true_binary*mask_bin, y_pred_binary*mask_bin)
    binary_loss = tf.keras.losses.binary_crossentropy(y_true_binary, y_pred_binary)
    
    # Calcular la pérdida total como la suma ponderada de las pérdidas
    total_loss = weighted_mae_loss + (2*binary_loss)
    
    return total_loss




def custom_loss_MAE_Cross(y_true, y_pred):
    # Obtener las dimensiones de los tensores de entrada
    num_quantitative_features = 4
    #num_binary_features = 1

    # Dividir las predicciones y las etiquetas verdaderas en partes cuantitativas y binarias
    y_true_quantitative, y_true_binary = y_true[:, :num_quantitative_features], y_true[:, num_quantitative_features:]
    y_pred_quantitative, y_pred_binary = y_pred[:, :num_quantitative_features], y_pred[:, num_quantitative_features:]

    # # Calcular el error cuadrático medio para las características cuantitativas
    # mse_quantitative = K.mean(K.square(y_true_quantitative - y_pred_quantitative), axis=-1)

    # # Calcular la entropía cruzada binaria para la característica binaria
    # binary_crossentropy = K.mean(K.binary_crossentropy(y_true_binary, y_pred_binary), axis=-1)

    # # Ponderar las pérdidas y combinarlas
    # weighted_loss = mse_quantitative * 0.8 + binary_crossentropy * 0.2
    # Calcular la pérdida de las variables cuantitativas (MAE ponderado)
    mae_loss = tf.reduce_mean(tf.abs(y_pred_quantitative - y_true_quantitative), axis=-1)
    weighted_mae_loss = tf.reduce_sum(mae_loss, axis=-1)  # Sumar las pérdidas de las cuatro variables
    
    # Calcular la pérdida de la variable binaria (entropía cruzada binaria)
    binary_loss = tf.keras.losses.binary_crossentropy(y_true_binary, y_pred_binary)
    
    # Calcular la pérdida total como la suma ponderada de las pérdidas
    total_loss = weighted_mae_loss + binary_loss
    
    return total_loss

