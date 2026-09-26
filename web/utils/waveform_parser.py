import io
import pandas as pd
import numpy as np

def parse_waveform_bytes(file_bytes, filename="uploaded.csv"):
    """
    Parses waveform file bytes robustly.
    Returns: dict with (time, voltage, warnings, stats)
    """
    warnings = []
    df_raw = None
    
    for enc in ["utf-8-sig", "utf-8", "cp949", "latin1"]:
        for sep in [",", ";", "\t", " "]:
            try:
                buf = io.BytesIO(file_bytes)
                df_try = pd.read_csv(buf, sep=sep, header=None, encoding=enc)
                if df_try.shape[1] >= 2:
                    df_raw = df_try
                    break
            except Exception:
                continue
        if df_raw is not None:
            break

    if df_raw is None or df_raw.shape[1] < 2:
        return {
            "success": False,
            "error": "Failed to parse CSV file. Expected at least 2 numeric columns (Time, Voltage).",
            "warnings": ["File format unreadable or fewer than 2 columns."]
        }

    first_row = df_raw.iloc[0]
    def is_float(x):
        try:
            float(x)
            return True
        except (ValueError, TypeError):
            return False

    has_header = not (is_float(first_row.iloc[0]) and is_float(first_row.iloc[1]))
    if has_header:
        df_data = df_raw.iloc[1:].copy()
        time_name_guess = str(df_raw.iloc[0, 0])
        volt_name_guess = str(df_raw.iloc[0, 1])
    else:
        df_data = df_raw.copy()
        time_name_guess = "Column 0"
        volt_name_guess = "Column 1"

    c0 = pd.to_numeric(df_data.iloc[:, 0], errors="coerce")
    c1 = pd.to_numeric(df_data.iloc[:, 1], errors="coerce")

    # Determine time axis vs voltage axis by monotonic increasing check
    c0_clean = c0.dropna()
    c1_clean = c1.dropna()

    diff0 = c0_clean.diff().fillna(0)
    diff1 = c1_clean.diff().fillna(0)

    mono0 = (diff0 > 0).sum()
    mono1 = (diff1 > 0).sum()

    if mono0 >= mono1:
        time_series = c0
        volt_series = c1
        t_name, v_name = time_name_guess, volt_name_guess
    else:
        time_series = c1
        volt_series = c0
        t_name, v_name = volt_name_guess, time_name_guess

    valid_mask = ~(time_series.isna() | volt_series.isna() | np.isinf(time_series) | np.isinf(volt_series))
    t_clean = time_series[valid_mask].to_numpy(dtype=float)
    v_clean = volt_series[valid_mask].to_numpy(dtype=float)

    n_points = len(t_clean)
    if n_points != 1024:
        warnings.append(f"INPUT FORMAT WARNING: Point count is {n_points}, expected standard 1024 points.")

    dt_diff = np.diff(t_clean)
    is_monotonic = bool(np.all(dt_diff > 0)) if len(dt_diff) > 0 else False
    if not is_monotonic:
        warnings.append("INPUT FORMAT WARNING: Time axis is not strictly monotonically increasing.")

    t_min = float(t_clean.min()) if n_points > 0 else 0.0
    t_max = float(t_clean.max()) if n_points > 0 else 0.0
    v_min = float(v_clean.min()) if n_points > 0 else 0.0
    v_max = float(v_clean.max()) if n_points > 0 else 0.0

    return {
        "success": True,
        "time": t_clean,
        "voltage": v_clean,
        "time_column": t_name,
        "voltage_column": v_name,
        "n_points": n_points,
        "time_min": t_min,
        "time_max": t_max,
        "voltage_min": v_min,
        "voltage_max": v_max,
        "is_monotonic": is_monotonic,
        "warnings": warnings
    }
