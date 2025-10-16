# src/func.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def convert_to_float(value):
    value = str(value).replace(',', '').replace('%', '')
    if 'K' in value: 
        return float(value.replace('K', '')) * 1_000
    elif 'M' in value: 
        return float(value.replace('M', '')) * 1_000_000
    elif 'B' in value:
        return float(value.replace('B', '')) * 1_000_000_000
    else: 
        return float(value)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)
    
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def calculate_metrics(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    accuracy_mape = 100 - mape

    print(f"\n--- Hasil Evaluasi {model_name} (Skala USD) ---")
    print(f"RMSE:     {rmse:,.2f}")
    print(f"MAE:      {mae:,.2f}")
    print(f"MAPE:     {mape:.2f}%")
    print(f"Akurasi (MAPE-based): {accuracy_mape:.2f}%")
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def calculate_metrics_scaled(y_true, y_pred, model_name="Model (Skala 0-1)"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"RMSE (pada returns): {rmse:.6f}")
    print(f"MAE (pada returns):  {mae:.6f}")
    
    return {"RMSE_scaled": rmse, "MAE_scaled": mae}
    
def inverse_transform_and_reconstruct(scaled_preds, scaler_obj, feature_list, price_data_for_recon):
    dummy_array = np.zeros((len(scaled_preds), len(feature_list)))
    dummy_array[:, 0] = scaled_preds.flatten()
    predicted_returns = scaler_obj.inverse_transform(dummy_array)[:, 0]
    
    reconstructed = []
    actuals_for_comparison = price_data_for_recon.iloc[1:].values.flatten()
    
    for i in range(len(predicted_returns)):
        last_known_price = price_data_for_recon.iloc[i]
        next_price_pred = last_known_price * (1 + predicted_returns[i])
        reconstructed.append(next_price_pred)
        
    return np.array(reconstructed), actuals_for_comparison