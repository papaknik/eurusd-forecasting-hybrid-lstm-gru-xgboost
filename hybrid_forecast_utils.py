
"""
hybrid_forecast_utils.py

Author: Nikos Papakostas
Description: Core functions for EUR/USD hybrid forecasting using LSTM-GRU + XGBoost.
Includes data preparation, hybrid modeling, evaluation, and rolling window backtesting.
"""

# =============================================================================
# Imports and Global Configurations
# =============================================================================
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from xgboost import XGBRegressor
import random

# setting the parameters for the sliding window
# These parameters can be adjusted based on the specific use case
window_size = 30
horizon = 1

# =============================================================================
# Data Preprocessing Functions
# =============================================================================

def generate_sliding_windows(series, window_size, horizon, target_col=0):
    """
    Transforms a univariate or multivariate time series into supervised learning format
    using a sliding window approach.

    Parameters:
        series (ndarray): Time series data with shape (n_samples, n_features).
        window_size (int): Number of time steps in each input window (lookback period).
        horizon (int): Forecast horizon — how many steps ahead to predict.
        target_col (int, optional): Index of the target variable in the multivariate series (default: 0).

    Returns:
        X (ndarray): 3D array of shape (num_samples, window_size, n_features) for model input.
        y (ndarray): 2D array of shape (num_samples, horizon) containing forecast targets.
    """
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size:i+window_size+horizon, target_col])
    return np.array(X), np.array(y)


def scale_and_cast(X_train, X_test, y_train, y_test):
    """
    Scales features and targets using MinMaxScaler and casts to float32 for TensorFlow compatibility.

    Parameters:
        X_train (ndarray): Training features (3D).
        X_test (ndarray): Test features (3D).
        y_train (ndarray): Training targets (1D).
        y_test (ndarray): Test targets (1D).

    Returns:
        X_train_scaled (ndarray): Scaled and float32-cast training features.
        X_test_scaled (ndarray): Scaled and float32-cast test features.
        y_train_scaled (ndarray): Scaled and float32-cast training targets.
        y_test_scaled (ndarray): Scaled and float32-cast test targets.
        y_scaler (MinMaxScaler): Target scaler for inverse transformation.
    """
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Flatten and scale, then reshape back to original
    X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape).astype(np.float32)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler

# =============================================================================
# Evaluation Metrics
# =============================================================================

def daily_directional_accuracy(y_true, y_pred):
    """
    Calculates the directional accuracy (DA) between predicted and actual values.
    DA measures how often the predicted direction (up/down) matches the true direction.

    Parameters:
        y_true (ndarray): Array of actual target values (1D).
        y_pred (ndarray): Array of predicted target values (1D).

    Returns:
        float: Directional accuracy as a percentage (0–100).
    """
    return 100 * np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))


def plot_zoomed_predictions(y_true, y_pred, n_subplots=4):
    """
    Plots zoomed-in segments comparing actual and predicted values.

    This visualization helps examine model performance on shorter, focused sections
    of the test set by splitting it into evenly sized chunks and plotting each one separately.

    Parameters:
        y_true (ndarray): Actual target values (1D array).
        y_pred (ndarray): Predicted values from the model (1D array).
        n_subplots (int, optional): Number of equally-sized segments to plot (default: 4).

    Returns:
        None. Displays matplotlib subplots.
    """
    subplot_size = len(y_true) // n_subplots
    plt.figure(figsize=(16, 4 * n_subplots))

    for i in range(n_subplots):
        start = i * subplot_size
        end = (i + 1) * subplot_size if i < n_subplots - 1 else len(y_true)

        plt.subplot(n_subplots, 1, i + 1)
        plt.plot(range(start, end), y_true[start:end], label='Actual', linewidth=2)
        plt.plot(range(start, end), y_pred[start:end], '--', label='Hybrid Prediction', linewidth=2)
        plt.title(f"Zoomed-in View: Test Set (samples {start} to {end})")
        plt.xlabel("Time Index")
        plt.ylabel("Exchange Rate")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

# =============================================================================
# Modeling phase (LSTM-GRU + XGBoost)
# =============================================================================

def hybrid_model_predict(
    X_train_scaled, y_train_scaled,
    X_test_scaled, y_test,
    y_scaler,
    window_size, horizon
):
    """
    Trains and evaluates a hybrid forecasting model consisting of a stacked LSTM-GRU neural network
    followed by an XGBoost meta-learner to model residual errors.

    The function:
    - Trains a deep RNN to predict the next value(s) based on past observations.
    - Computes residuals from the RNN predictions.
    - Trains an XGBoost model on structured residual features.
    - Combines the base prediction and the residual correction to produce final outputs.

    Parameters:
        X_train_scaled (ndarray): Scaled training input features (3D array: samples × time steps × features).
        y_train_scaled (ndarray): Scaled training target values (1D or 2D depending on horizon).
        X_test_scaled (ndarray): Scaled test input features (3D).
        y_test (ndarray): True test set target values (not scaled).
        y_scaler (MinMaxScaler): Scaler object used for inverse transforming predictions.
        window_size (int): Length of the input sequence window.
        horizon (int): Forecast horizon (number of time steps predicted).

    Returns:
        y_final (ndarray): Final hybrid model predictions (RNN + residual correction).
        mae (float): Mean Absolute Error of final predictions.
        mape (float): Mean Absolute Percentage Error of final predictions.
        da (float): Directional Accuracy (%) of final predictions.
        rnn_model (Sequential): Trained base LSTM-GRU model.
        meta_learner (XGBRegressor): Trained XGBoost model for residual learning.
    """
    # Set seeds for reproducibility
    K.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- RNN Model ---
    rnn_model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(256, return_sequences=True, activation='tanh'),
        Dropout(0.1),
        GRU(256, activation='tanh'),
        Dropout(0.1),
        Dense(horizon)
    ])
    rnn_model.compile(optimizer='adamw', loss='mse')

    rnn_model.fit(
        X_train_scaled, y_train_scaled,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)
        ],
        shuffle=False
    )

    # --- Prediction ---
    y_pred_scaled = rnn_model.predict(X_test_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

    # --- Residual Learning ---
    residuals = y_test - y_pred
    prev_resid = np.append([0], residuals[:-1])
    pred_delta = np.append([0], np.diff(y_pred))

    features_df = pd.DataFrame({
        'predicted_lstm': y_pred,
        'prev_resid': prev_resid,
        'pred_delta': pred_delta
    })

    meta_learner = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=2,
        random_state=42
    )
    meta_learner.fit(features_df, residuals)
    y_final = y_pred + meta_learner.predict(features_df)

    # --- Evaluation ---
    mae = mean_absolute_error(y_test, y_final)
    mape = mean_absolute_percentage_error(y_test, y_final) * 100
    da = daily_directional_accuracy(y_test, y_final)

    return y_final, mae, mape, da, rnn_model, meta_learner

# =============================================================================
# Rolling Evaluation
# =============================================================================

def rolling_window_backtest(
    series,
    index,
    start_year=2007,
    end_year=2025, #non-inclusive
    train_years=8,
    window_size=30,
    horizon=1
):
    """
    Performs rolling-window backtesting using an RNN + XGBoost hybrid model.

    Parameters:
        series (np.ndarray): Univariate time series (shape: [n, 1]).
        index (pd.DatetimeIndex): Date index aligned with the series.
        start_year (int): Starting year of training.
        end_year (int): Exclusive upper bound year for testing.
        train_years (int): Number of years for each training window.
        window_size (int): Number of past time steps used in supervised learning transformation.
        horizon (int): Forecast horizon (steps ahead).

    Returns:
        pd.DataFrame: Backtest results for each fold, including MAE, MAPE, DA.
    """
    results = []

    # Filter data by year
    mask = (index.year >= start_year) & (index.year <= end_year) # Ensures you don’t operate on out-of-range years
    index = index[mask]

    for train_start in range(start_year, end_year - train_years):
        train_end = train_start + train_years - 1
        test_year = train_end + 1

        # Masks for time slicing
        train_mask = (index.year >= train_start) & (index.year <= train_end)
        test_mask = index.year == test_year

        train_series = series[train_mask]
        test_series = series[test_mask]
        full_series = np.concatenate([train_series, test_series])
        full_index = index[train_mask | test_mask]

        # Supervised learning data
        X, y = generate_sliding_windows(full_series, window_size, horizon)
        y = y.flatten()

        # Split point
        split_idx = len(train_series) - window_size - horizon + 1
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # Scale and cast to float32
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_and_cast(
            X_train, X_test, y_train, y_test)

        # Predict using hybrid model
        y_final, mae, mape, da, rnn_model, meta_learner = hybrid_model_predict(
            X_train_scaled, y_train_scaled,
            X_test_scaled, y_test,
            y_scaler,
            window_size, horizon
        )

        # Save results
        results.append({
            "Train Period": f"{train_start}–{train_end}",
            "Test Period": f"{test_year}",
            "Train Size": len(X_train),
            "Test Size": len(X_test),
            "MAE": mae,
            "MAPE": mape,
            "DA": da
        })

    return pd.DataFrame(results)