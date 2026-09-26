---
title: TDR AI Soil Property Estimation
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚡ TDR-AI Soil Property Estimation & Waveform Analysis Platform

Full-Waveform Time Domain Reflectometry (TDR) Geotechnical Property Estimation powered by a Canonical Multi-Task 1D-CNN Production Engine.

### Capabilities:
- **Joint 5-Target Estimation**: Gravimetric Water Content ($w$), Volumetric Water Content ($\theta_v$), Dry Density ($\rho_d$), Bulk Electrical Conductivity ($EC_b$), and Pore-Water Electrical Conductivity ($EC_w$).
- **Dual Reflection Physics Analysis**: Automatic dual-tangent detection of first reflection ($t_1$), handle trough, and end-of-probe reflection ($t_2$), with dielectric permittivity ($K_a$) calculation and signal attenuation warnings.
- **Reference Validation Suite**: Built-in verification against laboratory benchmark measurements (`데이터 결과.xlsx`) and 21 standard water/NaCl calibration waveforms.
- **Exporting**: Instant CSV export of interpreted soil parameters and reflection characteristics.
