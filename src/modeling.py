# src/modeling.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import os


from .func import calculate_metrics_scaled, calculate_metrics, inverse_transform_and_reconstruct

def build_and_run_model(X_train, y_train, X_val, y_val, X_test, y_test,
                        scaler_obj, feature_list, val_split_idx, seq_len, original_df, assets_dir='assets'):
    

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Grafik Loss Model LSTM')
    plt.xlabel('Epoch'); plt.ylabel('Loss (MSE Skala 0-1)'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(assets_dir, 'training_loss.png')) 
    plt.close()

    predicted_scaled = model.predict(X_test)
    scaled_results = calculate_metrics_scaled(y_test, predicted_scaled, model_name="LSTM (Skala 0-1)")
    
    start_index = val_split_idx + seq_len - 1
    price_data_for_recon = original_df['Harga Aktual (USD)'].iloc[start_index:].reset_index(drop=True)
    
    reconstructed_prices, actual_prices = inverse_transform_and_reconstruct(
        predicted_scaled, scaler_obj, feature_list, price_data_for_recon
    )
    
    results = calculate_metrics(actual_prices, reconstructed_prices, model_name="LSTM")
    
    plot_dates = original_df['Date'].iloc[-len(actual_prices):].reset_index(drop=True)
    plt.figure(figsize=(15, 8))
    plt.plot(plot_dates, actual_prices, color='red', label='Actual', linewidth=2.5)
    plt.plot(plot_dates, reconstructed_prices, color='green', linestyle='--', label='Predicted')
    plt.title('Actual vs Predicted (Test Set)')
    plt.xlabel('Date'); plt.ylabel('Price (USD)'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(assets_dir, 'lstm_predicted.png')) 
    plt.close()

    return results, reconstructed_prices, actual_prices, model