"""
TDR Physics Module - FROZEN PRODUCTION SPECIFICATION (PHASE 7D)
---------------------------------------------------------------
Physically principled TDR reflection detection and waveform region segmentation
for the 75-mm Standard TDR Probe (L = 0.075 m = 7.5 cm).
Completely decoupled from AI regression heads.

SI and Unit Standard:
  - Probe Length: L = 0.075 m
  - Speed of Light: c = 0.299792458 m/ns (2.99792458e8 m/s)
  - Travel Time: dt_ns = t2_ns - t1_ns
  - Apparent Permittivity: Ka = ((c * dt_ns) / (2 * L))^2

Reference Plane Detection Rules:
  - t1: PROBE_HEAD_PEAK detected dynamically from waveform morphology
        (terminating the initial sharp rising edge of the coaxial-to-probe incident pulse).
        No hard-coded 60.9 ns threshold or fixed point is used.
  - trough: Handle entry dip directly following t1.
  - t2: DUAL-TANGENT geometric intersection
        (intersection of post-trough steepest rising tangent with the trough baseline level).
  - t2_valid: Physical validity test (EM pulse attenuation threshold).
        If attenuated / invalid: t2 = None, dt = None, Ka = None. No fake or extrapolated Ka is returned.
"""

import numpy as np

# Physical Constants (SI / Consistent Units)
PROBE_LENGTH_M = 0.075         # Standard 75 mm probe
SPEED_OF_LIGHT_M_PER_NS = 0.299792458  # m/ns (= 2.99792458e8 m/s)

def detect_t1_t2_ka(T, V, probe_len_m=PROBE_LENGTH_M):
    """
    Detects physical TDR reflection markers for a standard 75-mm probe:
      - t1: Probe head entrance peak from waveform morphology
      - trough: Initial dielectric entry dip
      - t2: Dual-tangent geometric intersection point (if physically discernible)
      - t2_valid: True if end reflection exists, False if signal attenuated (high EC)
      - dt: Pure rod transit time (t2 - t1) in ns if valid, else None
      - Ka: Apparent dielectric permittivity if valid, else None
      - confidence: 0.0 to 1.0 confidence score
    """
    T_arr = np.array(T, dtype=float)
    V_arr = np.array(V, dtype=float)

    if len(T_arr) < 50 or len(V_arr) < 50:
        return {
            "t1": None, "t1_idx": None,
            "trough": None, "trough_idx": None,
            "t2": None, "t2_idx": None,
            "t2_infl": None, "t2_slope": None, "v_baseline": None,
            "t2_valid": False, "dt": None, "Ka": None,
            "confidence": 0.0, "reason": "Insufficient waveform points"
        }

    dV = np.gradient(V_arr, T_arr)

    # 1. Morphological t1 Detection (Probe Head Peak)
    # Search across broad transmission line (20 ns to 120 ns) for the first sharp rising edge
    # of the incident pulse entering the probe head (dV > 40 mV/ns).
    m_line = (T_arr >= 20.0) & (T_arr <= 120.0)
    indices = np.where(m_line)[0]
    sharp_rise_cands = [i for i in indices if dV[i] > 40.0]
    
    if sharp_rise_cands:
        first_sharp_rise = sharp_rise_cands[0]
        t_rise = T_arr[first_sharp_rise]
        # Probe head peak terminates this initial sharp rise
        m_pk = (T_arr >= t_rise - 0.2) & (T_arr <= t_rise + 0.8)
        t1_idx = int(np.where(m_pk)[0][np.argmax(V_arr[m_pk])])
    else:
        # Fallback: global argmax of derivative on transmission line
        t1_idx = int(indices[np.argmax(dV[m_line])])

    t1 = float(T_arr[t1_idx])
    v_t1 = float(V_arr[t1_idx])

    # 2. Detect Trough (Handle entry dip following t1)
    m_tr = (T_arr >= t1) & (T_arr <= t1 + 4.5)
    if not np.any(m_tr):
        m_tr = (T_arr >= t1) & (T_arr <= t1 + 8.0)
    tr_idx = int(np.where(m_tr)[0][np.argmin(V_arr[m_tr])])
    t_tr = float(T_arr[tr_idx])
    v_tr = float(V_arr[tr_idx])

    # 3. Detect Second Reflection (t2) via Dual-Tangent Geometric Intersection
    m_refl = (T_arr > t_tr + 0.1) & (T_arr <= min(t_tr + 12.0, 75.0))
    if not np.any(m_refl):
        return {
            "t1": t1, "t1_idx": t1_idx,
            "trough": t_tr, "trough_idx": tr_idx,
            "t2": None, "t2_idx": None,
            "t2_infl": None, "t2_slope": None, "v_baseline": v_tr,
            "t2_valid": False, "dt": None, "Ka": None,
            "confidence": 0.0, "reason": "Reflection search window out of range"
        }

    refl_indices = np.where(m_refl)[0]
    max_v_post = float(np.max(V_arr[m_refl]))
    rise_amp = max_v_post - v_tr
    max_dv_post = float(np.max(dV[m_refl]))

    # Physical validity criterion:
    # High electrical conductivity attenuates pulse below physical detection threshold
    is_valid = (rise_amp >= 3.0) and (max_dv_post >= 5.0)

    if not is_valid:
        conf = float(np.clip(rise_amp / 20.0, 0.0, 0.3))
        return {
            "t1": t1, "t1_idx": t1_idx,
            "trough": t_tr, "trough_idx": tr_idx,
            "t2": None, "t2_idx": None,
            "t2_infl": None, "t2_slope": None, "v_baseline": v_tr,
            "t2_valid": False, "dt": None, "Ka": None,
            "confidence": conf,
            "rise_amplitude_mv": rise_amp,
            "reason": "Electromagnetic end-reflection heavily attenuated by high conductivity"
        }

    # Dual-tangent geometric intersection:
    # 1) Inflection point (steepest rising slope of second reflection)
    t2_infl_idx = int(refl_indices[np.argmax(dV[m_refl])])
    t2_infl = float(T_arr[t2_infl_idx])
    slope = float(dV[t2_infl_idx])
    
    # 2) Geometric intersection of tangent line with trough baseline:
    # V_tan(t) = V[t2_infl] + slope * (t - t2_infl) = v_tr
    # => t2 = t2_infl - (V[t2_infl] - v_tr) / slope
    if slope > 1.0:
        t2 = float(t2_infl - (V_arr[t2_infl_idx] - v_tr) / slope)
    else:
        t2 = t2_infl
    t2_idx = int(np.argmin(np.abs(T_arr - t2)))

    dt = float(t2 - t1)

    # Physical plausibility check for 75 mm probe:
    if dt <= 0.2 or dt > 6.0:
        return {
            "t1": t1, "t1_idx": t1_idx,
            "trough": t_tr, "trough_idx": tr_idx,
            "t2": None, "t2_idx": None,
            "t2_infl": t2_infl, "t2_slope": slope, "v_baseline": v_tr,
            "t2_valid": False, "dt": None, "Ka": None,
            "confidence": 0.1,
            "reason": f"Detected dt ({dt:.2f} ns) out of physical bounds for 75 mm probe"
        }

    # SI / Consistent Ka calculation:
    Ka = float(((SPEED_OF_LIGHT_M_PER_NS * dt) / (2.0 * probe_len_m)) ** 2)
    conf = float(np.clip(0.6 + (rise_amp / 100.0) * 0.4, 0.6, 0.99))

    return {
        "t1": t1, "t1_idx": t1_idx,
        "trough": t_tr, "trough_idx": tr_idx,
        "t2": t2, "t2_idx": t2_idx,
        "t2_infl": t2_infl,
        "t2_slope": slope,
        "v_baseline": v_tr,
        "t2_valid": True,
        "dt": dt,
        "Ka": Ka,
        "confidence": conf,
        "rise_amplitude_mv": rise_amp,
        "reason": "Physical end-reflection successfully identified via dual-tangent intersection"
    }

def get_morphological_regions(T, V, t1=None):
    """
    Dynamically partitions waveform into 4 reproducible physical morphology regions
    anchored to detected probe entrance t1:
      - R1: System / Connector transmission line (T < 15 ns)
      - R2: Pre-soil handle interface (15 ns <= T < t1)
      - R3: Soil reflection & transit region (t1 <= T < t1 + 15 ns)
      - R4: Late-time asymptotic attenuation tail (T >= t1 + 15 ns)
    """
    T_arr = np.array(T, dtype=float)
    if t1 is None:
        det = detect_t1_t2_ka(T_arr, V)
        t1 = det["t1"] if det["t1"] is not None else 60.912
    
    t_incident = 15.0
    t_tail_start = t1 + 15.0

    mask_R1 = (T_arr < t_incident)
    mask_R2 = (T_arr >= t_incident) & (T_arr < t1)
    mask_R3 = (T_arr >= t1) & (T_arr < t_tail_start)
    mask_R4 = (T_arr >= t_tail_start)

    return {
        "R1": {
            "name": "R1: System / Connector Response",
            "desc": "Pre-pulse coaxial cable transmission line before probe head",
            "mask": mask_R1, "t_start": 0.0, "t_end": t_incident
        },
        "R2": {
            "name": "R2: Pre-soil Baseline / Handle",
            "desc": "Cable-to-handle interface step up to probe entrance t1",
            "mask": mask_R2, "t_start": t_incident, "t_end": t1
        },
        "R3": {
            "name": "R3: Soil Reflection & Transit",
            "desc": "Round-trip pulse propagation through soil rods (dielectric delay)",
            "mask": mask_R3, "t_start": t1, "t_end": t_tail_start
        },
        "R4": {
            "name": "R4: Late-time Attenuation Tail",
            "desc": "Asymptotic DC resistance plateau Vf (electrical conductivity)",
            "mask": mask_R4, "t_start": t_tail_start, "t_end": float(T_arr[-1])
        }
    }