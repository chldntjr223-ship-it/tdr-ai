import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WATER_DIR = os.path.join(ROOT_DIR, "data", "raw", "water_nacl_reference")

_WATER_DATA = None

def get_water_reference_data():
    global _WATER_DATA
    if _WATER_DATA is not None:
        return _WATER_DATA
        
    ordered_files = ["공기.csv", "물.csv"] + [f"{i}.csv" for i in range(1, 20)]
    
    waveforms = {}
    table_rows = []
    
    for fn in ordered_files:
        fp = os.path.join(WATER_DIR, fn)
        if not os.path.exists(fp):
            # Check ascii fallbacks
            if fn == "공기.csv":
                fp = os.path.join(WATER_DIR, "air.csv")
            elif fn == "물.csv":
                fp = os.path.join(WATER_DIR, "water.csv")
            if not os.path.exists(fp):
                continue
            
        df = pd.read_csv(fp, header=None)
        c0 = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
        c1 = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
        if np.mean(np.diff(c0) >= 0) >= np.mean(np.diff(c1) >= 0):
            t, v = c0, c1
        else:
            t, v = c1, c0
            
        label = "Air (공기)" if fn == "공기.csv" else ("Distilled Water (순수 증류수)" if fn == "물.csv" else f"NaCl Step #{fn.replace('.csv', '')}")
        
        v_init = float(v[0])
        v_final = float(np.mean(v[-100:]))
        
        peak_idx = int(np.argmax(v[:500]))
        v_t1 = float(v[peak_idx])
        t_t1 = float(t[peak_idx])
        
        tr_start = max(peak_idx, 300)
        trough_idx = tr_start + int(np.argmin(v[tr_start:700]))
        v_trough = float(v[trough_idx])
        t_trough = float(t[trough_idx])
        
        refl_search = v[trough_idx:900]
        if len(refl_search) > 0:
            refl_max = float(np.max(refl_search))
            refl_height = refl_max - v_trough
        else:
            refl_height = 0.0
            
        waveforms[fn] = {
            "filename": fn,
            "label": label,
            "time": t,
            "voltage": v,
            "v_initial": v_init,
            "v_final": v_final,
            "v_trough": v_trough,
            "t_trough": t_trough,
            "v_t1": v_t1,
            "t_t1": t_t1,
            "refl_height": refl_height
        }
        
        table_rows.append({
            "Sequence": fn,
            "Condition": label,
            "Points": len(t),
            "V_initial (mV)": round(v_init, 1),
            "V_t1 (mV)": round(v_t1, 1),
            "V_trough (mV)": round(v_trough, 1),
            "T_trough (ns)": round(t_trough, 2),
            "V_final (mV)": round(v_final, 1),
            "Refl Height (mV)": round(refl_height, 1)
        })
        
    _WATER_DATA = {
        "waveforms": waveforms,
        "table": pd.DataFrame(table_rows),
        "ordered_files": ordered_files
    }
    return _WATER_DATA
