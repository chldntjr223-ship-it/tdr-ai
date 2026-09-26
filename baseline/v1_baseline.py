import os
import json
import math
import numpy as np

KB_PATH = os.path.join(os.path.dirname(__file__), "master_kb.json")

def load_knowledge_base():
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

_MASTER_KB = None

def get_kb():
    global _MASTER_KB
    if _MASTER_KB is None:
        _MASTER_KB = load_knowledge_base()
    return _MASTER_KB

def surgical_scout(T, V, is_dcpt=False):
    T_arr = np.array(T, dtype=float)
    V_arr = np.array(V, dtype=float)

    if len(T_arr) < 10 or len(V_arr) < 10:
        return {
            "t1_idx": 0, "t2_idx": 0,
            "Vn": V_arr.tolist(),
            "V0": 0.0, "t1": 0.0, "t2": 0.0, "b_base": 0.0
        }

    skip_val = 18
    skip_idx = np.argmax(T_arr > skip_val)
    if skip_idx == 0 and T_arr[0] <= skip_val:
        skip_idx = len(T_arr) - 1

    departure_idx = -1
    for i in range(skip_idx, len(T_arr) - 5):
        if abs(V_arr[i + 5] - V_arr[i - 5]) > 15:
            departure_idx = i
            break
    if departure_idx == -1:
        departure_idx = skip_idx

    t1_idx = departure_idx
    has_risen = False
    for i in range(departure_idx, len(T_arr) - 2):
        if V_arr[i + 1] > V_arr[i] + 0.5:
            has_risen = True
        if has_risen and V_arr[i + 1] < V_arr[i]:
            t1_idx = i
            break
        if T_arr[i] > T_arr[departure_idx] + 20:
            break

    if t1_idx == -1:
        t1_idx = 0

    # t2: Anchor to absolute bottom of trough after t1 (Legacy t2)
    t2_idx = t1_idx + 1
    min_v = float("inf")
    search_limit = min(t1_idx + 150, len(T_arr) - 1)

    for i in range(t1_idx + 1, search_limit):
        if V_arr[i] < min_v:
            min_v = V_arr[i]
            t2_idx = i
        if i > t1_idx + 5 and V_arr[i] > min_v + 1.5:
            break

    b_base = float(np.mean(V_arr[:50])) if len(V_arr) >= 50 else 0.0
    Vn = (V_arr - b_base).tolist()

    return {
        "t1_idx": int(t1_idx),
        "t2_idx": int(t2_idx),
        "Vn": Vn,
        "V0": float(V_arr[t1_idx] - b_base),
        "t1": float(T_arr[t1_idx]),
        "t2": float(T_arr[t2_idx]),
        "b_base": b_base
    }

REF_INDEX_PATH = os.path.join(os.path.dirname(__file__), "reference_index.json")

def load_reference_index():
    if os.path.exists(REF_INDEX_PATH):
        with open(REF_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

_REF_INDEX = None

def get_ref_index():
    global _REF_INDEX
    if _REF_INDEX is None:
        _REF_INDEX = load_reference_index()
    return _REF_INDEX

def run_legacy_baseline(
    T,
    V,
    probe_length_cm=7.5,
    leave_one_out=False,
    exclude_sample_id=None,
    exclude_sha256=None,
    uploaded_sha256=None,
    exclude_path=None,
    exclude_filename=None,
    exclude_soil=None,
    **kwargs
):
    """
    Runs the Legacy V1 baseline estimation.
    If leave_one_out is True:
      Excludes self-reference using strict priority:
        1. Exact SHA-256 match
        2. Exact unique sample_id match
        3. Exact normalized full-path match
      Distance and basename are NOT used to exclude non-self references.
    """
    scout = surgical_scout(T, V, is_dcpt=False)
    dt = scout["t2"] - scout["t1"]
    
    vn_arr = np.array(scout["Vn"])
    vf_sub = float(np.mean(vn_arr[-200:])) if len(vn_arr) >= 200 else float(np.mean(vn_arr))
    vf_real = vf_sub + scout["b_base"]

    soil_eps = math.pow(dt / (2.0 * probe_length_cm / 30.0), 2)
    std_dt = dt

    # Resolve exclusion identifiers
    effective_sha256 = (exclude_sha256 or uploaded_sha256 or "").lower().strip()
    effective_sid = int(exclude_sample_id) if exclude_sample_id is not None else None
    effective_path = os.path.normpath(exclude_path).lower() if exclude_path else None

    # If any unique identifier is provided, enforce leave_one_out mode
    if effective_sid is not None or effective_sha256 or effective_path:
        leave_one_out = True

    kb = get_kb()
    ref_idx = get_ref_index()
    best = None
    min_d = float("inf")

    for k_i, k in enumerate(kb):
        d_dt = math.pow((k["dt"] - std_dt) / 0.5, 2)
        d_ec = math.pow((k["vf"] - vf_sub) / 30.0, 2)
        total_d = d_dt + d_ec

        if leave_one_out:
            ref_meta = ref_idx[k_i] if (ref_idx and k_i < len(ref_idx)) else {}
            ref_sha = (ref_meta.get("sha256") or "").lower().strip()
            ref_sid = ref_meta.get("sample_id")
            ref_path = os.path.normpath(ref_meta.get("csv_path") or "").lower() if ref_meta.get("csv_path") else None

            is_self = False
            # Priority 1: Exact SHA-256 match
            if effective_sha256 and ref_sha and effective_sha256 == ref_sha:
                is_self = True
            # Priority 2: Exact unique sample_id match
            elif effective_sid is not None and ref_sid is not None and effective_sid == ref_sid:
                is_self = True
            # Priority 3: Exact normalized full-path match
            elif effective_path and ref_path and effective_path == ref_path:
                is_self = True

            if is_self:
                continue

        if total_d < min_d:
            min_d = total_d
            best = k
            best_idx = k_i

    if best is not None and best_idx is not None:
        best_meta = ref_idx[best_idx] if (ref_idx and best_idx < len(ref_idx)) else {}
        ref_sample_id = best_meta.get("sample_id")
        ref_sha256 = best_meta.get("sha256")
        ref_csv_path = best_meta.get("csv_path")
        w_pred = float(best.get("w", 0.0))
        rho_pred = float(best.get("d", 0.0))
        sigma_mix_pred = float(best.get("eb", 0.0))
        sigma_w_pred = float(best.get("ew", best.get("eb", 0.0)))
        theta_v_pred = (w_pred / 100.0) * rho_pred / 1.0
        soil_type = best.get("soil", "Unknown")
        ref_filename = best.get("filename", "")
    else:
        best_idx = None
        ref_sample_id = None
        ref_sha256 = None
        ref_csv_path = None
        w_pred = float("nan")
        rho_pred = float("nan")
        sigma_mix_pred = float("nan")
        sigma_w_pred = float("nan")
        theta_v_pred = float("nan")
        soil_type = "None"
        ref_filename = ""

    return {
        "dt": dt,
        "t1": scout["t1"],
        "t2": scout["t2"],
        "t1_idx": scout["t1_idx"],
        "t2_idx": scout["t2_idx"],
        "Vf": vf_real,
        "Ka": soil_eps,
        "V0": scout["V0"],
        "Vn": scout["Vn"],
        "w": w_pred,
        "rho_d": rho_pred,
        "theta_v": theta_v_pred,
        "sigma_mix": sigma_mix_pred,
        "sigma_w": sigma_w_pred,
        "ref_soil": soil_type,
        "ref_filename": ref_filename,
        "ref_kb_index": best_idx,
        "ref_sample_id": ref_sample_id,
        "ref_sha256": ref_sha256,
        "ref_csv_path": ref_csv_path,
        "match_distance": min_d
    }
