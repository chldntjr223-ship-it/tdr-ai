import os
import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "tdr_multitask_cnn_v8_corrected_labels.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, "data", "final_model_v8", "tdr_multitask_cnn_v8_corrected_labels.pth")

SCALER_PATH = os.path.join(ROOT_DIR, "scalers", "target_standard_scaler_train_only_corrected.pkl")
if not os.path.exists(SCALER_PATH):
    SCALER_PATH = os.path.join(ROOT_DIR, "data", "dataset_candidate_a_v8", "target_standard_scaler_train_only_corrected.pkl")

CONFIG_PATH = os.path.join(ROOT_DIR, "models", "model_config_v8.json")

TARGET_COLS = ["theta_v", "w_percent", "rho_d_gcm3", "ecb_sm", "ecw_sm"]
TARGET_LABELS = {
    "theta_v": "θv (m3/m3)",
    "w_percent": "w (%)",
    "rho_d_gcm3": "ρd (g/cm3)",
    "ecb_sm": "ECb (S/m)",
    "ecw_sm": "ECw (S/m)"
}

UNIFORM_TIME_GRID = np.linspace(0.0, 150.0, 1024)

# Canonical 1D-CNN Multi-Task Architecture (5 Conv1D Blocks + 3 Linear Layers)
class TDRMultiTaskCNN(nn.Module):
    def __init__(self, n_outputs=5):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1: 1 -> 16
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 2: 16 -> 32
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 3: 32 -> 64
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 4: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 5: 128 -> 128
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        out = self.regressor(feat)
        return out

# Global Singleton Caches
_V8_MODEL = None
_V8_SCALER = None

def get_v8_model():
    global _V8_MODEL
    if _V8_MODEL is None:
        model = TDRMultiTaskCNN(n_outputs=len(TARGET_COLS))
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        _V8_MODEL = model
    return _V8_MODEL

def get_v8_scaler():
    global _V8_SCALER
    if _V8_SCALER is None:
        _V8_SCALER = joblib.load(SCALER_PATH)
    return _V8_SCALER

def preprocess_waveform(time_array, voltage_array):
    """
    Standard preprocessing matching V8 training:
    1. Sort by time
    2. Interpolate linearly to uniform 1024 points (0.0 to 150.0 ns)
    3. Min-Max normalization to [0, 1] per waveform
    """
    t = np.asarray(time_array, dtype=float)
    v = np.asarray(voltage_array, dtype=float)
    
    # Valid numeric check
    valid_mask = ~(np.isnan(t) | np.isnan(v) | np.isinf(t) | np.isinf(v))
    t = t[valid_mask]
    v = v[valid_mask]
    
    # Sort by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    v = v[sort_idx]
    
    # Interpolate
    f_interp = interp1d(t, v, kind='linear', fill_value='extrapolate')
    v_1024 = f_interp(UNIFORM_TIME_GRID)
    
    # Min-Max normalization per waveform
    v_min = float(v_1024.min())
    v_max = float(v_1024.max())
    v_norm = (v_1024 - v_min) / (v_max - v_min + 1e-12)
    
    return UNIFORM_TIME_GRID, v_1024, v_norm.astype(np.float32)

def predict_v8_multitask(time_array, voltage_array):
    """
    Runs V8 Multi-Task 1D-CNN inference on raw time and voltage arrays.
    Returns: dict with unscaled predictions, raw values, and preprocessed waveform.
    """
    t_grid, v_interp, v_norm = preprocess_waveform(time_array, voltage_array)
    
    # Reshape for 1D-CNN: (batch=1, channel=1, length=1024)
    x_tensor = torch.tensor(v_norm[np.newaxis, np.newaxis, :], dtype=torch.float32)
    
    model = get_v8_model()
    scaler = get_v8_scaler()
    
    with torch.no_grad():
        raw_scaled_pred = model(x_tensor).numpy() # shape: (1, 5)
        
    unscaled_pred = scaler.inverse_transform(raw_scaled_pred)[0] # shape: (5,)
    
    res = {
        "theta_v": float(unscaled_pred[0]),
        "w_percent": float(unscaled_pred[1]),
        "rho_d_gcm3": float(unscaled_pred[2]),
        "ecb_sm": float(unscaled_pred[3]),
        "ecw_sm": float(unscaled_pred[4]),
        # UI / legacy-compatible aliases
        "w": float(unscaled_pred[1]),
        "rho_d": float(unscaled_pred[2]),
        "sigma_mix": float(unscaled_pred[3]),
        "sigma_w": float(unscaled_pred[4]),
        "raw_unscaled_vector": unscaled_pred.copy(),
        "t_grid": t_grid,
        "v_interp": v_interp,
        "v_norm": v_norm
    }
    return res
