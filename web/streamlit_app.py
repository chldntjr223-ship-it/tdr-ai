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

from baseline.v1_baseline import run_legacy_baseline
from physics.tdr_physics import detect_t1_t2_ka
from web.utils.waveform_parser import parse_waveform_bytes
from web.utils.ground_truth_matcher import match_ground_truth
from web.utils.ai_infer import predict_v8_multitask, TARGET_LABELS
from web.utils.water_reference_loader import get_water_reference_data

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

st.set_page_config(
    page_title="TDR-AI Soil Property Estimation",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ TDR-AI Soil Property Estimation & Waveform Analysis Platform")
st.markdown("Full-Waveform Time Domain Reflectometry Analysis — **Canonical Multi-Task 1D-CNN Production Engine**")

# Sidebar Configuration
st.sidebar.title("⚙️ System Configuration")
st.sidebar.markdown("**Probe Type:** Standard TDR Probe (3-Rod)")
st.sidebar.info("📏 **Probe Length:** `75 mm (7.5 cm fixed)`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Active AI Engine:**")
st.sidebar.success("🚀 `Multi-Task 1D-CNN (Corrected V8)`")
st.sidebar.caption("Inputs: 1-Ch, 1024-pt [0, 1] Normalized Waveform (0–150 ns)")
st.sidebar.caption("Outputs: 5 Joint Parameters (w, θv, ρd, ECb, ECw)")
st.sidebar.markdown("---")
st.sidebar.markdown("**Ground Truth Authority:**")
st.sidebar.caption("📖 `데이터 결과.xlsx` (Standard Probe Physical Benchmark)")
st.sidebar.caption("Consistent Formula: θv = w · ρd / 100")

# Top Navigation Tabs
tab_soil, tab_validation, tab_water, tab_provenance = st.tabs([
    "⚡ Soil Property AI Estimation",
    "📊 Model Validation (Test Split N=48)",
    "🧪 Water & Ionic Solution Reference",
    "ℹ️ System & Provenance"
])

# ==============================================================================
# TAB 1: SOIL PROPERTY AI ESTIMATION
# ==============================================================================
with tab_soil:
    st.subheader("📤 Upload TDR Waveform")
    uploaded_file = st.file_uploader(
        "Upload a 2-column CSV file (Time [ns], Voltage [mV])",
        type=["csv", "CSV"],
        help="Select or drag-and-drop a raw TDR waveform CSV file."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        # Calculate SHA-256 ONCE at entry
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()

        # 1. Parse Waveform
        parse_res = parse_waveform_bytes(file_bytes, filename)
        if not parse_res["success"]:
            st.error(parse_res["error"])
            st.stop()

        T = parse_res["time"]
        V = parse_res["voltage"]

        # 2. Run Canonical Corrected V8 Multi-Task 1D-CNN Inference ONCE
        ai_pred = predict_v8_multitask(T, V)
        gt = match_ground_truth(file_sha256)

        if gt is not None:
            display_mode = "Experimental Reference"
            displayed_w = float(gt["w"])
            displayed_theta_v = float(gt["theta_v"])
            displayed_rho_d = float(gt["rho_d"])
            displayed_ecb = float(gt["ecb"])
            displayed_ecw = float(gt["ecw"])
        else:
            display_mode = "AI Prediction"
            displayed_w = float(ai_pred["w_percent"])
            displayed_theta_v = float(ai_pred["theta_v"])
            displayed_rho_d = float(ai_pred["rho_d_gcm3"])
            displayed_ecb = float(ai_pred["ecb_sm"])
            displayed_ecw = float(ai_pred["ecw_sm"])

        # Build single authoritative inference result object
        current_run = {
            "sha256": file_sha256,
            "filename": filename,
            "n_points": parse_res["n_points"],
            "time": T,
            "voltage": V,
            "display_mode": display_mode,
            # Top card displayed values:
            "displayed_w": displayed_w,
            "displayed_theta_v": displayed_theta_v,
            "displayed_rho_d": displayed_rho_d,
            "displayed_ecb": displayed_ecb,
            "displayed_ecw": displayed_ecw,
            # Raw AI predictions:
            "ai_w_percent": float(ai_pred["w_percent"]),
            "ai_theta_v": float(ai_pred["theta_v"]),
            "ai_rho_d_gcm3": float(ai_pred["rho_d_gcm3"]),
            "ai_ecb_sm": float(ai_pred["ecb_sm"]),
            "ai_ecw_sm": float(ai_pred["ecw_sm"]),
            # Compatibility aliases
            "w_percent": float(ai_pred["w_percent"]),
            "theta_v": float(ai_pred["theta_v"]),
            "rho_d_gcm3": float(ai_pred["rho_d_gcm3"]),
            "ecb_sm": float(ai_pred["ecb_sm"]),
            "ecw_sm": float(ai_pred["ecw_sm"]),
            "gt": gt
        }
        st.session_state["current_inference"] = current_run

        # Status Badge Determination
        if gt is not None:
            split_tag = str(gt.get("split", "reference")).lower()
            if split_tag == "test":
                status_badge = "🎯 Known Reference [Independent Test Set (Unseen)]"
                status_color = "#00aa00"
            elif split_tag in ["train", "val"]:
                status_badge = f"⚠️ Known Reference [In-Sample Internal {split_tag.upper()} Split]"
                status_color = "#ff9900"
            else:
                status_badge = "📋 Known Reference [Standard Probe Catalog]"
                status_color = "#0088cc"
        else:
            status_badge = "🌐 Public Inference Mode [External / Field Specimen]"
            status_color = "#666666"

        # 3. TOP SECTION: Primary Geotechnical Properties
        st.markdown("---")
        if gt is not None:
            top_title = "Experimental Reference Soil Properties"
        else:
            top_title = "Estimated Soil Properties"
        st.markdown(f"### 📋 {top_title}" if gt is not None else f"### 🎯 {top_title}")
        
        info_col1, info_col2, info_col3 = st.columns([2, 2, 2])
        info_col1.markdown(f"**📄 Uploaded Waveform:** `{filename}` ({parse_res['n_points']} pts)")
        info_col2.markdown(f"**🤖 Model Version:** `Corrected V8 Multi-Task 1D-CNN`")
        info_col3.markdown(f"**🔍 Inference Status:** <span style='color:{status_color}; font-weight:bold;'>{status_badge}</span>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💧 Gravimetric Water (w)", f"{current_run['displayed_w']:.3f} %")
        c2.metric("🌊 Volumetric Water (θv)", f"{current_run['displayed_theta_v']:.4f} m³/m³")
        c3.metric("🧱 Dry Density (ρd)", f"{current_run['displayed_rho_d']:.3f} g/cm³")
        c4.metric("⚡ Bulk EC (ECb)", f"{current_run['displayed_ecb']:.4f} S/m")
        c5.metric("🧪 Pore-Water EC (ECw)", f"{format_ecw(current_run['displayed_ecw'])} S/m")

        # Result Export Button (Dual columns for Known Reference)
        if gt is not None:
            export_df = pd.DataFrame([{
                "Filename": filename,
                "SHA-256": current_run["sha256"],
                "Display Mode": current_run["display_mode"],
                "Soil Type": gt.get("soil_type", "Unknown"),
                "Dataset Split": gt.get("split", "Reference"),
                # Experimental Reference Values
                "Reference Gravimetric Water w (%)": round(current_run["displayed_w"], 4),
                "Reference Volumetric Water θv (m3/m3)": round(current_run["displayed_theta_v"], 4),
                "Reference Dry Density ρd (g/cm3)": round(current_run["displayed_rho_d"], 4),
                "Reference Bulk EC ECb (S/m)": round(current_run["displayed_ecb"], 4),
                "Reference Pore-Water EC ECw (S/m)": round(current_run["displayed_ecw"], 5),
                # AI Predicted Values (V8 Multi-Task CNN)
                "AI Predicted w (%)": round(current_run["ai_w_percent"], 4),
                "AI Predicted θv (m3/m3)": round(current_run["ai_theta_v"], 4),
                "AI Predicted ρd (g/cm3)": round(current_run["ai_rho_d_gcm3"], 4),
                "AI Predicted ECb (S/m)": round(current_run["ai_ecb_sm"], 4),
                "AI Predicted ECw (S/m)": round(current_run["ai_ecw_sm"], 4),
                # Absolute Errors
                "Absolute Error w (%)": round(abs(current_run["ai_w_percent"] - current_run["displayed_w"]), 4),
                "Absolute Error θv (m3/m3)": round(abs(current_run["ai_theta_v"] - current_run["displayed_theta_v"]), 4),
                "Absolute Error ρd (g/cm3)": round(abs(current_run["ai_rho_d_gcm3"] - current_run["displayed_rho_d"]), 4),
                "Absolute Error ECb (S/m)": round(abs(current_run["ai_ecb_sm"] - current_run["displayed_ecb"]), 4),
                "Absolute Error ECw (S/m)": round(abs(current_run["ai_ecw_sm"] - current_run["displayed_ecw"]), 5),
                "Model Version": "Corrected V8 Multi-Task 1D-CNN (Fixed 75-mm Probe)"
            }])
        else:
            export_df = pd.DataFrame([{
                "Filename": filename,
                "SHA-256": current_run["sha256"],
                "Display Mode": current_run["display_mode"],
                "AI Predicted Gravimetric Water w (%)": round(current_run["ai_w_percent"], 4),
                "AI Predicted Volumetric Water θv (m3/m3)": round(current_run["ai_theta_v"], 4),
                "AI Predicted Dry Density ρd (g/cm3)": round(current_run["ai_rho_d_gcm3"], 4),
                "AI Predicted Bulk EC ECb (S/m)": round(current_run["ai_ecb_sm"], 4),
                "AI Predicted Pore-Water EC ECw (S/m)": round(current_run["ai_ecw_sm"], 4),
                "Model Version": "Corrected V8 Multi-Task 1D-CNN (Fixed 75-mm Probe)"
            }])
        csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_bytes,
            file_name=f"tdr_{'reference' if gt else 'prediction'}_{os.path.splitext(filename)[0]}.csv",
            mime="text/csv"
        )

        # 4. Waveform Reflection Visualization (Physics-based)
        st.markdown("---")
        st.markdown("### 📈 TDR Waveform & Reflection Morphology")
        phys = detect_t1_t2_ka(T, V, probe_len_m=0.075)

        fig, ax = plt.subplots(figsize=(10, 3.8), facecolor="white")
        ax.plot(T, V, color="#1f77b4", linewidth=1.5, label="Raw Voltage V(t)")

        if phys["t1"] is not None:
            v_t1 = V[phys["t1_idx"]] if phys["t1_idx"] is not None else V[0]
            ax.plot(phys["t1"], v_t1, 'o', color="#0055ff", markersize=7, label=f"t₁ (Entrance): {phys['t1']:.2f} ns")
            ax.axvline(phys["t1"], color="#0055ff", linestyle=":", alpha=0.5)

        if phys["trough"] is not None:
            v_tr = V[phys["trough_idx"]] if phys["trough_idx"] is not None else V[0]
            ax.plot(phys["trough"], v_tr, 'v', color="#9900cc", markersize=7, label=f"Trough (Handle Dip): {phys['trough']:.2f} ns")

        if phys["t2_valid"] and phys["t2"] is not None:
            v_base = phys.get("v_baseline", V[phys["trough_idx"]])
            t2_val = phys["t2"]
            t2_infl = phys.get("t2_infl", t2_val)
            slope = phys.get("t2_slope", 10.0)

            t_base_span = np.linspace(phys["trough"], min(t2_val + 2.0, float(T[-1])), 30)
            ax.plot(t_base_span, np.full_like(t_base_span, v_base), color="#9900cc", linestyle=":", linewidth=1.2, alpha=0.7, label="Trough Baseline")

            t_tan_span = np.linspace(max(t2_val - 1.0, float(T[0])), min(t2_infl + 2.0, float(T[-1])), 30)
            v_tan = v_base + slope * (t_tan_span - t2_val)
            ax.plot(t_tan_span, v_tan, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Rising Tangent ({slope:.1f} mV/ns)")

            ax.plot(t2_val, v_base, 's', color="#00aa00", markersize=8, label=f"t₂ (Intersection): {t2_val:.2f} ns")
            ax.axvline(t2_val, color="#00aa00", linestyle="--", alpha=0.6)

        ax.set_xlabel("Time (ns)", fontsize=10)
        ax.set_ylabel("Voltage (mV)", fontsize=10)
        status_tag = f"[t₂ VALID | Ka = {phys['Ka']:.1f}]" if phys["t2_valid"] else "[⚠️ t₂ INVALID | Attenuated]"
        ax.set_title(f"Waveform: {filename} ({parse_res['n_points']} pts) — {status_tag}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        if not phys["t2_valid"]:
            st.warning(f"⚠️ **Reflection Pulse Attenuated**: High conductivity dissipates the rod-end reflection pulse ({phys['reason']}).")

        # 5. AI Prediction Comparison & Ground Truth Verification Section
        st.markdown("---")
        st.markdown("### 📊 AI Prediction Comparison")
        gt = current_run["gt"]

        if gt is not None:
            split_tag = str(gt.get("split", "reference")).lower()
            
            if split_tag in ["train", "val"]:
                st.warning(
                    f"⚠️ **In-Sample Reference Comparison (Internal {split_tag.upper()} Split)**\n\n"
                    f"This specimen is part of the model's internal training/validation dataset (Soil: **{gt['soil_type']}**). "
                    f"The table below reflects **reference fitting consistency**, **NOT an unseen out-of-sample generalization test**."
                )
            elif split_tag == "test":
                st.success(
                    f"✅ **Independent Test Set Specimen (Unseen Condition)**\n\n"
                    f"This waveform belongs to the **strict 48-sample Group Split Test set** (Soil: **{gt['soil_type']}**). "
                    f"The errors below reflect genuine out-of-sample generalization."
                )
            else:
                st.info(
                    f"📋 **Standard Probe Catalog Reference**\n\n"
                    f"Matched against experimental benchmark (`데이터 결과.xlsx` | Soil: **{gt['soil_type']}**)."
                )

            st.markdown("##### 🤖 Corrected V8 1D-CNN Model Predictions")
            c_ai1, c_ai2, c_ai3, c_ai4, c_ai5 = st.columns(5)
            c_ai1.metric("V8 Pred w", f"{current_run['ai_w_percent']:.3f} %")
            c_ai2.metric("V8 Pred θv", f"{current_run['ai_theta_v']:.4f} m³/m³")
            c_ai3.metric("V8 Pred ρd", f"{current_run['ai_rho_d_gcm3']:.3f} g/cm³")
            c_ai4.metric("V8 Pred ECb", f"{current_run['ai_ecb_sm']:.4f} S/m")
            c_ai5.metric("V8 Pred ECw", f"{current_run['ai_ecw_sm']:.4f} S/m")

            # Reference Comparison Table - Strictly tied to current_run
            gt_table = [
                {"Physical Property": "Gravimetric Water Content w (%)", "Ground Truth Reference": f"{gt['w']:.3f} %", "V8 Prediction": f"{current_run['ai_w_percent']:.3f} %", "Absolute Error": f"{abs(current_run['ai_w_percent'] - gt['w']):.3f} %", "Relative Error (%)": f"{abs(current_run['ai_w_percent'] - gt['w'])/max(1e-4, abs(gt['w']))*100:.1f} %"},
                {"Physical Property": "Volumetric Water Content θv (m³/m³)", "Ground Truth Reference": f"{gt['theta_v']:.4f}", "V8 Prediction": f"{current_run['ai_theta_v']:.4f}", "Absolute Error": f"{abs(current_run['ai_theta_v'] - gt['theta_v']):.4f}", "Relative Error (%)": f"{abs(current_run['ai_theta_v'] - gt['theta_v'])/max(1e-4, abs(gt['theta_v']))*100:.1f} %"},
                {"Physical Property": "Dry Density ρd (g/cm³)", "Ground Truth Reference": f"{gt['rho_d']:.3f} g/cm³", "V8 Prediction": f"{current_run['ai_rho_d_gcm3']:.3f} g/cm³", "Absolute Error": f"{abs(current_run['ai_rho_d_gcm3'] - gt['rho_d']):.3f} g/cm³", "Relative Error (%)": f"{abs(current_run['ai_rho_d_gcm3'] - gt['rho_d'])/max(1e-4, abs(gt['rho_d']))*100:.1f} %"},
                {"Physical Property": "Bulk EC ECb (S/m)", "Ground Truth Reference": f"{gt['ecb']:.4f} S/m", "V8 Prediction": f"{current_run['ai_ecb_sm']:.4f} S/m", "Absolute Error": f"{abs(current_run['ai_ecb_sm'] - gt['ecb']):.4f} S/m", "Relative Error (%)": f"{abs(current_run['ai_ecb_sm'] - gt['ecb'])/max(1e-4, abs(gt['ecb']))*100:.1f} %"},
                {"Physical Property": "Pore-Water EC ECw (S/m)", "Ground Truth Reference": f"{format_ecw(gt['ecw'])} S/m", "V8 Prediction": f"{current_run['ai_ecw_sm']:.4f} S/m", "Absolute Error": f"{abs(current_run['ai_ecw_sm'] - gt['ecw']):.4f} S/m", "Relative Error (%)": f"{abs(current_run['ai_ecw_sm'] - gt['ecw'])/max(1e-4, abs(gt['ecw']))*100:.1f} %"}
            ]
            st.table(pd.DataFrame(gt_table))
        else:
            st.info("🌐 **Public Inference Mode**: External/Field waveform detected. Multi-task predictions generated without ground truth comparison.")

        # 6. Auxiliary Modules (Collapsible)
        st.markdown("---")
        with st.expander("🏛️ Auxiliary 1: Legacy V1 Baseline (Leave-One-Out Nearest Reference)", expanded=False):
            base_loo = run_legacy_baseline(
                T, V,
                probe_length_cm=7.5,
                leave_one_out=True,
                uploaded_sha256=current_run["sha256"],
                exclude_filename=filename,
                exclude_soil=gt["soil_type"] if gt else None
            )
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("V1 w", f"{base_loo['w']:.3f} %")
            b2.metric("V1 θv", f"{base_loo['theta_v']:.4f}")
            b3.metric("V1 ρd", f"{base_loo['rho_d']:.3f} g/cm³")
            b4.metric("V1 ECb", f"{base_loo['sigma_mix']:.4f} S/m")
            b5.metric("V1 ECw", f"{base_loo['sigma_w']:.4f} S/m")
            st.caption(f"LOO Reference: `{base_loo['ref_soil']}` / `{base_loo['ref_filename']}` | Feature Distance: `{base_loo['match_distance']:.4f}`")

        with st.expander("📐 Auxiliary 2: Electromagnetic Reflection Parameters & Ka", expanded=False):
            st.write(phys)

        with st.expander("📊 Auxiliary 3: Preprocessed 1,024-pt Min-Max Waveform (AI Input Tensor)", expanded=False):
            fig_p, ax_p = plt.subplots(figsize=(9, 2.8))
            ax_p.plot(ai_pred["t_grid"], ai_pred["v_norm"], color="#2ca02c", linewidth=1.2)
            ax_p.set_title("Preprocessed Waveform fed to 1D-CNN (1024 Uniform Points, Min-Max [0, 1])", fontsize=10)
            ax_p.set_xlabel("Time (ns)", fontsize=9)
            ax_p.set_ylabel("Normalized Voltage [0, 1]", fontsize=9)
            ax_p.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig_p)
            plt.close(fig_p)
    else:
        st.info("👆 Please upload a TDR waveform CSV file to begin analysis.")

# ==============================================================================
# TAB 2: MODEL VALIDATION SUMMARY (INDEPENDENT TEST SET N=48)
# ==============================================================================
with tab_validation:
    st.subheader("📊 Model Validation Summary: Independent Test Split (N=48)")
    st.markdown("""
    This section summarizes the rigorous out-of-sample generalization performance of the **Corrected V8 Multi-Task 1D-CNN** 
    on the **48 independent Test split waveforms** (stratified Group Split by soil family with zero replicate leakage).
    """)

    st.markdown("### 1. Primary Physical Validation Metrics (Test N=48)")
    
    test_metrics = [
        {"Target Physical Property": "Volumetric Water Content θv (m³/m³)", "R²": "0.9535", "RMSE": "0.0554", "MAE": "0.0304", "MAPE (%) [Supplementary]": "16.50 %"},
        {"Target Physical Property": "Gravimetric Water Content w (%)", "R²": "0.9431", "RMSE": "13.3728", "MAE": "6.8216", "MAPE (%) [Supplementary]": "21.31 %"},
        {"Target Physical Property": "Dry Density ρd (g/cm³)", "R²": "0.9855", "RMSE": "0.0448", "MAE": "0.0329", "MAPE (%) [Supplementary]": "3.83 %"},
        {"Target Physical Property": "Bulk Electrical Conductivity ECb (S/m)", "R²": "0.9773", "RMSE": "0.0786", "MAE": "0.0485", "MAPE (%) [Supplementary]": "82.42 %"},
        {"Target Physical Property": "Pore-Water Electrical Conductivity ECw (S/m)", "R²": "0.9103", "RMSE": "0.3530", "MAE": "0.2524", "MAPE (%) [Supplementary]": "181.97 %"}
    ]
    st.table(pd.DataFrame(test_metrics))
    st.caption("📌 **Note on Metrics**: In low electrical conductivity conditions (e.g., ECb, ECw < 0.05 S/m), minor absolute deviations (±0.02 S/m) naturally generate mathematically large percentage ratios due to the near-zero denominator. RMSE and MAE are therefore presented as the primary error metrics, while MAPE is provided as a supplementary metric.")

    st.markdown("---")
    st.markdown("### 2. Multi-Task vs Single-Task Comparison")
    st.markdown("Comparison between the 5-target Multi-Task 1D-CNN and 5 individually trained Single-Task CNNs under identical split and hyperparameters:")

    st_mt_comp = [
        {"Target": "θv (m³/m³)", "Multi-Task R²": "0.9535", "Single-Task R²": "0.9384", "Δ R²": "+0.0151", "Multi-Task RMSE": "0.0554", "Single-Task RMSE": "0.0638", "Multi-Task MAE": "0.0304", "Single-Task MAE": "0.0392"},
        {"Target": "w (%)", "Multi-Task R²": "0.9431", "Single-Task R²": "0.8968", "Δ R²": "+0.0463", "Multi-Task RMSE": "13.3728", "Single-Task RMSE": "18.0102", "Multi-Task MAE": "6.8216", "Single-Task MAE": "11.0086"},
        {"Target": "ρd (g/cm³)", "Multi-Task R²": "0.9855", "Single-Task R²": "0.9757", "Δ R²": "+0.0098", "Multi-Task RMSE": "0.0448", "Single-Task RMSE": "0.0580", "Multi-Task MAE": "0.0329", "Single-Task MAE": "0.0439"},
        {"Target": "ECb (S/m)", "Multi-Task R²": "0.9773", "Single-Task R²": "0.9648", "Δ R²": "+0.0125", "Multi-Task RMSE": "0.0786", "Single-Task RMSE": "0.0979", "Multi-Task MAE": "0.0485", "Single-Task MAE": "0.0509"},
        {"Target": "ECw (S/m)", "Multi-Task R²": "0.9103", "Single-Task R²": "0.9127", "Δ R²": "-0.0024", "Multi-Task RMSE": "0.3530", "Single-Task RMSE": "0.3483", "Multi-Task MAE": "0.2524", "Single-Task MAE": "0.2438"}
    ]
    st.table(pd.DataFrame(st_mt_comp))
    st.info("💡 **Key Finding**: The shared feature representation in Multi-Task learning enforces dielectric-conductivity physical consistency, improving water content w R² by +0.0463 and reducing RMSE from 18.01 to 13.37.")

    st.markdown("---")
    st.markdown("### 3. CNN vs Conventional Models")
    st.markdown("Evaluation across Conventional range (w ≤ 35%, θv ≤ 0.45, ECw ≤ 1.0 S/m) and Extended range (w > 35% or θv > 0.45 or ECw > 1.0 S/m):")
    
    table2_comp = [
        {"Target": "w (%)", "Model": "Conventional (Yu & Drnevich)", "Conventional MAPE": "99.81 %", "Extended MAPE": "99.94 %", "Overall MAPE": "99.87 %", "Overall RMSE": "80.801"},
        {"Target": "w (%)", "Model": "1D-CNN (Corrected V8)", "Conventional MAPE": "31.11 %", "Extended MAPE": "9.74 %", "Overall MAPE": "21.31 %", "Overall RMSE": "13.373"},
        {"Target": "ρd (g/cm³)", "Model": "Conventional (Yu & Drnevich)", "Conventional MAPE": "73.56 %", "Extended MAPE": "265.73 %", "Overall MAPE": "161.64 %", "Overall RMSE": "1.429"},
        {"Target": "ρd (g/cm³)", "Model": "1D-CNN (Corrected V8)", "Conventional MAPE": "1.84 %", "Extended MAPE": "6.20 %", "Overall MAPE": "3.83 %", "Overall RMSE": "0.0448"},
        {"Target": "ECw (S/m)", "Model": "Conventional (Chen et al.)", "Conventional MAPE": "28,282.55 %", "Extended MAPE": "3,728.52 %", "Overall MAPE": "8,332.40 %", "Overall RMSE": "9.100"},
        {"Target": "ECw (S/m)", "Model": "1D-CNN (Corrected V8)", "Conventional MAPE": "361.42 %", "Extended MAPE": "140.55 %", "Overall MAPE": "181.97 %", "Overall RMSE": "0.3530"},
        {"Target": "ECb (S/m)", "Model": "Conventional (Giese-Tiemann)", "Conventional MAPE": "52.78 %", "Extended MAPE": "81.37 %", "Overall MAPE": "55.17 %", "Overall RMSE": "0.4816"},
        {"Target": "ECb (S/m)", "Model": "1D-CNN (Corrected V8)", "Conventional MAPE": "89.01 %", "Extended MAPE": "9.94 %", "Overall MAPE": "82.42 %", "Overall RMSE": "0.0786"},
        {"Target": "θv (m³/m³)", "Model": "Conventional (Topp et al.)", "Conventional MAPE": "69.10 %", "Extended MAPE": "84.01 %", "Overall MAPE": "75.94 %", "Overall RMSE": "0.4440"},
        {"Target": "θv (m³/m³)", "Model": "1D-CNN (Corrected V8)", "Conventional MAPE": "28.27 %", "Extended MAPE": "2.58 %", "Overall MAPE": "16.50 %", "Overall RMSE": "0.0554"}
    ]
    st.table(pd.DataFrame(table2_comp))

# ==============================================================================
# TAB 3: WATER & IONIC SOLUTION REFERENCE
# ==============================================================================
with tab_water:
    st.subheader("🧪 Water & Ionic Solution Reference Dataset")
    st.markdown("""
    This reference suite contains **21 baseline measurements** obtained with the Standard Probe:
    - **Air (`공기.csv`)**: Baseline dielectrics in open air ($K_a \\approx 1$, zero attenuation).
    - **Distilled Water (`물.csv`)**: Pure water reference ($K_a \\approx 80$, strong dielectric reflection, near-zero conductivity).
    - **NaCl Addition Steps #1 to #19 (`1.csv` ~ `19.csv`)**: Stepwise addition of NaCl to distilled water, capturing progressive ohmic signal dissipation, drop in steady-state voltage ($V_f$), and eventual rod-end reflection collapse.
    
    *(Note: Tracked as sequential experimental additions without speculative concentration values.)*
    """)
    
    water_data = get_water_reference_data()
    w_files = water_data["ordered_files"]
    w_waveforms = water_data["waveforms"]
    w_table = water_data["table"]
    
    view_preset = st.radio(
        "Display Preset:",
        ["Progression Steps (#1, #5, #10, #15, #19)", "Air vs Pure Water", "Complete Sequence Overlay (21 waveforms)", "Custom Multi-Select"],
        horizontal=True
    )
    
    if view_preset == "Progression Steps (#1, #5, #10, #15, #19)":
        selected_seqs = ["공기.csv", "물.csv", "1.csv", "5.csv", "10.csv", "15.csv", "19.csv"]
    elif view_preset == "Air vs Pure Water":
        selected_seqs = ["공기.csv", "물.csv"]
    elif view_preset == "Complete Sequence Overlay (21 waveforms)":
        selected_seqs = w_files
    else:
        selected_seqs = st.multiselect("Select waveforms to display:", w_files, default=["공기.csv", "물.csv", "1.csv", "10.csv", "19.csv"])
        
    if selected_seqs:
        fig_w, (ax_w1, ax_w2) = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Color mapping
        colormap = plt.cm.plasma(np.linspace(0, 1, len(selected_seqs)))
        
        for idx, fn in enumerate(selected_seqs):
            if fn in w_waveforms:
                item = w_waveforms[fn]
                t_w = item["time"]
                v_w = item["voltage"]
                c = "#000000" if fn=="공기.csv" else ("#0055ff" if fn=="물.csv" else colormap[idx])
                ax_w1.plot(t_w, v_w, label=item["label"], color=c, linewidth=1.5 if fn in ["공기.csv", "물.csv"] else 1.2)
                
        ax_w1.set_title("TDR Waveform Overlay Across Ionic Concentration Range", fontsize=11, fontweight="bold")
        ax_w1.set_xlabel("Time (ns)", fontsize=10)
        ax_w1.set_ylabel("Reflection Voltage (mV)", fontsize=10)
        ax_w1.grid(True, linestyle=":", alpha=0.6)
        ax_w1.legend(loc="upper right", fontsize=8, ncol=2)
        ax_w1.set_xlim(0, 150)
        
        # Plot 2: Steady state final voltage Vf trend
        nacl_num_files = [f"{i}.csv" for i in range(1, 20)]
        steps = list(range(1, 20))
        vf_trend = [w_waveforms[fn]["v_final"] for fn in nacl_num_files]
        trough_trend = [w_waveforms[fn]["v_trough"] for fn in nacl_num_files]
        
        ax_w2.plot(steps, vf_trend, 'o-', color="#d62728", linewidth=2, label="Steady-State Final Voltage Vf (mV)")
        ax_w2.plot(steps, trough_trend, 's--', color="#1f77b4", linewidth=1.5, label="Trough Voltage (mV)")
        ax_w2.axhline(w_waveforms["물.csv"]["v_final"], color="#0055ff", linestyle=":", label="Pure Water Baseline (+256.0 mV)")
        ax_w2.set_title("Monotonic Drop in Steady-State Voltage (Vf) vs NaCl Step #1–#19", fontsize=11, fontweight="bold")
        ax_w2.set_xlabel("NaCl Sequence Step (1 -> 19)", fontsize=10)
        ax_w2.set_ylabel("Voltage (mV)", fontsize=10)
        ax_w2.grid(True, linestyle=":", alpha=0.6)
        ax_w2.set_xticks(steps)
        ax_w2.legend(loc="upper right", fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)
        
    st.markdown("#### 📋 Water & NaCl Physical Parameter Summary Table")
    st.dataframe(w_table, use_container_width=True)

# ==============================================================================
# TAB 4: SYSTEM & MODEL PROVENANCE
# ==============================================================================
with tab_provenance:
    st.subheader("ℹ️ System Architecture & Model Provenance")
    st.markdown("""
    ### 1. Canonical Multi-Task 1D-CNN Specifications
    - **Backbone Architecture**: 5 Sequential Conv1D Blocks (Channel Filters: 16 -> 32 -> 64 -> 128 -> 128)
      - Each block equipped with 1D Batch Normalization, ReLU activation, and MaxPool1D(kernel=2, stride=2).
    - **Regression Head**: Flatten (4096) -> Linear(4096, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 64) -> ReLU -> **Linear(64, 5)**
    - **Training Pipeline**:
      - 319 Uniform 0–150 ns Full Waveforms (1,024 points).
      - **Dataset Composition**: Total 250 distinct experimental conditions across 6 soil families.
        - **Train Split**: 223 waveforms from **174 experimental conditions** (69.6% of conditions).
        - **Validation Split**: 48 waveforms from **37 experimental conditions** (14.8% of conditions).
        - **Test Split**: 48 waveforms from **39 experimental conditions** (15.6% of conditions).
      - **Target Scaling**: StandardScaler fit strictly on the 223 Train samples only.
      - **Ground Truth Authority**: Fixed to `데이터 결과.xlsx` (100% matched, 0 mismatches across 5 targets).
    """)
