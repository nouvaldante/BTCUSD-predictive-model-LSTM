# main.py

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

# Import from src
from src.func import convert_to_float, create_sequences, calculate_metrics
from src.modeling import build_and_run_model


# Setup and initialization

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

DATA_PATH = 'dataset/Bitcoin Historical Data.csv'
MODEL_SAVE_PATH = 'models/model_bitcoin_lstm.h5'
ASSETS_DIR = 'assets'

os.makedirs('models', exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 30
FEATURES = ['Price_Returns', 'sinyal_bb_beli', 'sinyal_macd_beli']
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1

def run_pipeline():
   

    # 1. Data loading/preprocessing
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATA_PATH}.")
        return

    df.rename(columns={'Price': 'Harga Aktual (USD)'}, inplace=True)
    cols_to_convert = ['Harga Aktual (USD)', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    for col in cols_to_convert:
        df[col] = df[col].apply(convert_to_float) 
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Feature Engineering
    window_bb = 20
    df['bb_middle'] = df['Harga Aktual (USD)'].rolling(window=window_bb).mean()
    df['bb_std'] = df['Harga Aktual (USD)'].rolling(window=window_bb).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    ema_12 = df['Harga Aktual (USD)'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Harga Aktual (USD)'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['Price_Returns'] = df['Harga Aktual (USD)'].pct_change()
    df['sinyal_bb_beli'] = np.where(df['Harga Aktual (USD)'] <= df['bb_lower'], 1, 0)
    df['sinyal_macd_beli'] = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) < df['macd_signal'].shift(1)), 1, 0)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # Data Scaling and Sequencing
    data_for_modeling = df[FEATURES].astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    data_scaled = scaler.fit_transform(data_for_modeling)
    X, y = create_sequences(data_scaled, SEQUENCE_LENGTH)

    # Data Splitting
    train_split_point = int(TRAIN_RATIO * len(X))
    val_split_point = int((TRAIN_RATIO + VAL_RATIO) * len(X))
    X_train, y_train = X[:train_split_point], y[:train_split_point]
    X_val, y_val = X[train_split_point:val_split_point], y[train_split_point:val_split_point]
    X_test, y_test = X[val_split_point:], y[val_split_point:]
    


    # 2. Model Training dan Evaluation

    lstm_results, reconstructed_prices, actual_prices, final_lstm_model = build_and_run_model(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        scaler_obj=scaler, feature_list=FEATURES,
        val_split_idx=val_split_point, seq_len=SEQUENCE_LENGTH,
        original_df=df, assets_dir=ASSETS_DIR
    )


    # 3. Save model

    final_lstm_model.save(MODEL_SAVE_PATH)
    print(f"\n--- Model Disimpan: {MODEL_SAVE_PATH} ---")
    
    print(f"RMSE (USD): {lstm_results['RMSE']:,.2f}")
    print(f"MAE (USD):  {lstm_results['MAE']:,.2f}")
    print(f"MAPE (%):   {lstm_results['MAPE']:.2f}%")
    
if __name__ == "__main__":
    run_pipeline()