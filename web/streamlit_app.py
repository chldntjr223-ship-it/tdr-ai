import os
import sys
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.tdr_physics import detect_t1_t2_ka
from web.utils.waveform_parser import parse_waveform_bytes
from web.utils.ground_truth_matcher import match_ground_truth
from web.utils.ai_infer import predict_v8_multitask

# Page configuration
st.set_page_config(
    page_title="TDR-AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Balanced 2-Column Minimal Engineering Dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }
    
    /* Main container max-width 1500px and centered */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1500px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Slim sidebar styling (200px) */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        width: 200px !important;
        min-width: 200px !important;
        max-width: 200px !important;
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebarContent"], section[data-testid="stSidebar"] > div {
        width: 200px !important;
        min-width: 200px !important;
        padding: 1.5rem 0.8rem !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Compact File Uploader */
    [data-testid="stFileUploader"] {
        margin-bottom: 1.2rem;
    }
    [data-testid="stFileUploader"] section {
        padding: 0.7rem 1.2rem !important;
        border: 1px dashed #cbd5e1;
        background-color: #ffffff;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #2563eb;
    }
    
    /* Result Card Styling */
    .prop-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        height: 80px;
        box-sizing: border-box;
    }
    .prop-card:hover {
        border-color: #cbd5e1;
    }
    
    /* Clean Download Button */
    .stDownloadButton > button {
        background-color: #0f172a;
        color: #ffffff;
        border: 1px solid #0f172a;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        height: 44px;
        width: 100%;
        transition: all 0.15s ease-in-out;
        margin-top: 6px;
    }
    .stDownloadButton > button:hover {
        background-color: #1e293b;
        border-color: #1e293b;
        color: #ffffff;
    }
    
    /* Responsive adjustment: on screens <= 1024px, ensure natural 1-column stack */
    @media (max-width: 1024px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 1.5rem !important;
        }
        [data-testid="stColumn"], [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

def format_ecw(val):
    if val is None:
        return "N/A"
    s = f"{val:.5f}".rstrip('0')
    if s.endswith('.'):
        s = s[:-1]
    parts = s.split('.')
    if len(parts) == 2 and len(parts[1]) < 4:
        s = f"{val:.4f}"
    return s

# Lucide-style SVG Icons (28px)
ICON_WATER_DROP = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg>'
ICON_WATER_WAVES = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12c2.5-3 5.5-3 8 0s5.5 3 8 0 5.5-3 8 0"></path><path d="M2 17c2.5-3 5.5-3 8 0s5.5 3 8 0 5.5-3 8 0"></path></svg>'
ICON_DENSITY_CUBE = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>'
ICON_BOLT = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'
ICON_FLASK = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31"></path><path d="M14 2v7.31"></path><path d="M8.5 2h7"></path><path d="M14 9.3a6.5 6.5 0 1 1-4 0"></path><path d="M5.52 16h12.96"></path></svg>'

def render_result_card(icon_svg, title, value_str, unit_str):
    card_html = (
        f'<div class="prop-card">'
        f'<div style="display:flex; align-items:center; gap:14px;">'
        f'<div style="display:flex; align-items:center; justify-content:center; width:40px; height:40px; '
        f'border-radius:8px; background:#eff6ff; flex-shrink:0;">'
        f'{icon_svg}'
        f'</div>'
        f'<div style="font-size:0.95rem; font-weight:600; color:#1e293b; line-height:1.2;">{title}</div>'
        f'</div>'
        f'<div style="text-align:right; flex-shrink:0; margin-left:12px;">'
        f'<span style="font-size:1.5rem; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">{value_str}</span>'
        f'<span style="font-size:0.85rem; font-weight:500; color:#64748b; margin-left:5px;">{unit_str}</span>'
        f'</div>'
        f'</div>'
    )
    return card_html

# ==============================================================================
# 1. SIDEBAR (Minimal Read-Only System Information)
# ==============================================================================
with st.sidebar:
    st.markdown("""
<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin-bottom:12px;">
    System Specs
</div>
<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:10px 12px; margin-bottom:10px;">
    <div style="font-size:0.7rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:2px;">Probe Length</div>
    <div style="font-size:1.05rem; font-weight:700; color:#0f172a;">75 mm</div>
</div>
<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:10px 12px;">
    <div style="font-size:0.7rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:2px;">AI Engine</div>
    <div style="font-size:0.86rem; font-weight:700; color:#0f172a; white-space:nowrap;">Multi-Task 1D-CNN</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MAIN PAGE HEADER (Compact)
# ==============================================================================
st.markdown("""
<div style="display:flex; align-items:baseline; gap:16px; margin-bottom:12px;">
    <h1 style="font-size:2.1rem; font-weight:800; color:#0f172a; margin:0; letter-spacing:-0.02em;">TDR-AI</h1>
    <span style="font-size:1.05rem; color:#64748b; font-weight:400;">Full-Waveform Soil Property Estimation</span>
</div>
""", unsafe_allow_html=True)

# CSV File Uploader (Compact)
uploaded_file = st.file_uploader(
    "Upload TDR Waveform CSV",
    type=["csv", "CSV"],
    label_visibility="collapsed",
    help="Select or drag-and-drop a raw 2-column TDR waveform CSV file (Time [ns], Voltage [mV])."
)

# ==============================================================================
# 3. 2-COLUMN BALANCED ANALYSIS AREA
# Left: Waveform Graph (13:17 aspect ratio, ~500px x 650px)
# Right: 5 Result Cards + Download Button (~440px x 650px)
# ==============================================================================
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()

    # 1. Parse Waveform Bytes
    parse_res = parse_waveform_bytes(file_bytes, filename)
    if not parse_res["success"]:
        st.error(parse_res["error"])
        st.stop()

    T = parse_res["time"]
    V = parse_res["voltage"]

    # 2. Run Inference & Match Reference ONCE
    ai_pred = predict_v8_multitask(T, V)
    gt = match_ground_truth(file_sha256)

    # Determine Display Values
    if gt is not None:
        display_mode = "Experimental Reference"
        badge_text = "Experimental Reference"
        badge_bg = "#dbeafe"
        badge_fg = "#1e40af"
        badge_border = "#bfdbfe"
        disp_w = float(gt["w"])
        disp_th = float(gt["theta_v"])
        disp_rho = float(gt["rho_d"])
        disp_ecb = float(gt["ecb"])
        disp_ecw = float(gt["ecw"])
    else:
        display_mode = "AI Prediction"
        badge_text = "AI Estimate"
        badge_bg = "#f1f5f9"
        badge_fg = "#475569"
        badge_border = "#e2e8f0"
        disp_w = float(ai_pred["w_percent"])
        disp_th = float(ai_pred["theta_v"])
        disp_rho = float(ai_pred["rho_d_gcm3"])
        disp_ecb = float(ai_pred["ecb_sm"])
        disp_ecw = float(ai_pred["ecw_sm"])

    # 2-Column Balanced Container: 60% Graph / 40% Results
    col_graph, col_results = st.columns([60, 40], gap="large")

    # --------------------------------------------------------------------------
    # LEFT COLUMN: Waveform Graph (Landscape ~17:13 / 4:3 Aspect Ratio)
    # --------------------------------------------------------------------------
    with col_graph:
        phys = detect_t1_t2_ka(T, V, probe_len_m=0.075)

        # Prominent Landscape Waveform Graph (width ~730px, height ~550px)
        fig, ax = plt.subplots(figsize=(7.3, 5.5), facecolor="white")
        ax.plot(T, V, color="#1d4ed8", linewidth=2.2, label="Waveform V(t)")

        # Reflection markers
        if phys["t1"] is not None:
            v_t1 = V[phys["t1_idx"]] if phys["t1_idx"] is not None else V[0]
            ax.plot(phys["t1"], v_t1, 'o', color="#2563eb", markersize=7, label=f"t₁ (Entrance): {phys['t1']:.1f} ns")
            ax.axvline(phys["t1"], color="#2563eb", linestyle=":", alpha=0.55)

        if phys["t2_valid"] and phys["t2"] is not None:
            v_base = phys.get("v_baseline", V[phys["trough_idx"]]) if phys.get("trough_idx") is not None else V[0]
            t2_val = phys["t2"]
            ax.plot(t2_val, v_base, 's', color="#059669", markersize=7, label=f"t₂ (Reflection): {t2_val:.1f} ns")
            ax.axvline(t2_val, color="#059669", linestyle="--", alpha=0.55)

        ax.set_title("TDR Reflection Waveform", fontsize=12.5, fontweight="bold", color="#0f172a", pad=10)
        ax.set_xlabel("Time (ns)", fontsize=10.5, fontweight="500", color="#334155", labelpad=6)
        ax.set_ylabel("Voltage (mV)", fontsize=10.5, fontweight="500", color="#334155", labelpad=6)
        ax.tick_params(colors="#64748b", labelsize=9.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#cbd5e1")
        ax.spines['bottom'].set_color("#cbd5e1")
        ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        ax.legend(loc="upper right", fontsize=9.2, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.95)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------------------------
    # RIGHT COLUMN: 5 Result Cards + Download Button
    # --------------------------------------------------------------------------
    with col_results:
        # Header + Badge
        header_html = (
            f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; height:34px;">'
            f'<div style="font-size:1.2rem; font-weight:700; color:#0f172a; letter-spacing:-0.01em;">'
            f'Soil Physical Properties'
            f'</div>'
            f'<span style="background:{badge_bg}; color:{badge_fg}; font-size:0.75rem; font-weight:600; '
            f'padding:4px 12px; border-radius:9999px; border:1px solid {badge_border}; letter-spacing:0.02em;">'
            f'{badge_text}'
            f'</span>'
            f'</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)

        # Card 1: Gravimetric Water Content (w)
        st.markdown(render_result_card(
            ICON_WATER_DROP,
            "Gravimetric Water Content (w)",
            f"{disp_w:.3f}",
            "%"
        ), unsafe_allow_html=True)

        # Card 2: Volumetric Water Content (θv)
        st.markdown(render_result_card(
            ICON_WATER_WAVES,
            "Volumetric Water Content (θv)",
            f"{disp_th:.4f}",
            "m³/m³"
        ), unsafe_allow_html=True)

        # Card 3: Dry Density (ρd)
        st.markdown(render_result_card(
            ICON_DENSITY_CUBE,
            "Dry Density (ρd)",
            f"{disp_rho:.3f}",
            "g/cm³"
        ), unsafe_allow_html=True)

        # Card 4: Bulk Electrical Conductivity (ECb)
        st.markdown(render_result_card(
            ICON_BOLT,
            "Bulk Electrical Conductivity (ECb)",
            f"{disp_ecb:.4f}",
            "S/m"
        ), unsafe_allow_html=True)

        # Card 5: Pore-Water Electrical Conductivity (ECw)
        st.markdown(render_result_card(
            ICON_FLASK,
            "Pore-Water Electrical Conductivity (ECw)",
            f"{format_ecw(disp_ecw)}",
            "S/m"
        ), unsafe_allow_html=True)

        # Download Results Button
        if gt is not None:
            export_df = pd.DataFrame([{
                "Filename": filename,
                "SHA-256": file_sha256,
                "Display Mode": display_mode,
                "Soil Type": gt.get("soil_type", "Unknown"),
                "Dataset Split": gt.get("split", "Reference"),
                # Reference Values
                "Reference Gravimetric Water w (%)": round(disp_w, 4),
                "Reference Volumetric Water θv (m3/m3)": round(disp_th, 4),
                "Reference Dry Density ρd (g/cm3)": round(disp_rho, 4),
                "Reference Bulk EC ECb (S/m)": round(disp_ecb, 4),
                "Reference Pore-Water EC ECw (S/m)": round(disp_ecw, 5),
                # AI Predictions
                "AI Predicted w (%)": round(float(ai_pred["w_percent"]), 4),
                "AI Predicted θv (m3/m3)": round(float(ai_pred["theta_v"]), 4),
                "AI Predicted ρd (g/cm3)": round(float(ai_pred["rho_d_gcm3"]), 4),
                "AI Predicted ECb (S/m)": round(float(ai_pred["ecb_sm"]), 4),
                "AI Predicted ECw (S/m)": round(float(ai_pred["ecw_sm"]), 4),
                # Errors
                "Absolute Error w (%)": round(abs(float(ai_pred["w_percent"]) - disp_w), 4),
                "Absolute Error θv (m3/m3)": round(abs(float(ai_pred["theta_v"]) - disp_th), 4),
                "Absolute Error ρd (g/cm3)": round(abs(float(ai_pred["rho_d_gcm3"]) - disp_rho), 4),
                "Absolute Error ECb (S/m)": round(abs(float(ai_pred["ecb_sm"]) - disp_ecb), 4),
                "Absolute Error ECw (S/m)": round(abs(float(ai_pred["ecw_sm"]) - disp_ecw), 5),
                "Model Version": "Multi-Task 1D-CNN (Fixed 75-mm Probe)"
            }])
        else:
            export_df = pd.DataFrame([{
                "Filename": filename,
                "SHA-256": file_sha256,
                "Display Mode": display_mode,
                "AI Predicted Gravimetric Water w (%)": round(float(ai_pred["w_percent"]), 4),
                "AI Predicted Volumetric Water θv (m3/m3)": round(float(ai_pred["theta_v"]), 4),
                "AI Predicted Dry Density ρd (g/cm3)": round(float(ai_pred["rho_d_gcm3"]), 4),
                "AI Predicted Bulk EC ECb (S/m)": round(float(ai_pred["ecb_sm"]), 4),
                "AI Predicted Pore-Water EC ECw (S/m)": round(float(ai_pred["ecw_sm"]), 4),
                "Model Version": "Multi-Task 1D-CNN (Fixed 75-mm Probe)"
            }])

        csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="Download Results (CSV)",
            data=csv_bytes,
            file_name=f"tdr_{'reference' if gt else 'prediction'}_{os.path.splitext(filename)[0]}.csv",
            mime="text/csv"
        )
else:
    # Idle empty state when no file is uploaded yet
    st.markdown("""
<div style="
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    padding: 36px 20px;
    text-align: center;
    background: #ffffff;
    margin-top: 14px;
">
    <div style="font-size: 0.95rem; font-weight: 500; color: #475569; margin-bottom: 4px;">
        Select or drop a TDR waveform CSV file to begin analysis
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8;">
        Standard format: 2 columns (Time [ns], Voltage [mV])
    </div>
</div>
""", unsafe_allow_html=True)
