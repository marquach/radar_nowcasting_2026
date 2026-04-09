# Precipitation Nowcasting with MeteoSwiss Radar Data

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-blue.svg)](https://pytorch.org)
[![pysteps](https://img.shields.io/badge/pysteps-v1.8-orange.svg)](https://pysteps.github.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-brightgreen.svg)](https://streamlit.io)

**Deep Learning for short-term precipitation forecasting (0-30min) using real MeteoSwiss RZC radar data.** 

## 🎯 Project Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| **1. Data Pipeline** | ✅ Complete | MCH RZC data → PyTorch Dataset with sequences  |
| **2. Visualization** | ✅ Complete | pyvis animations of radar sequences |
| **3. U-Net Training** | 🔄 In Progress | Conv3D U-Net for 1-step nowcasting |
| **4. Multi-Step + CSI** | ⏳ Planned | 3/6-step predictions + weather metrics  |
| **5. Baseline Comparison** | ⏳ Planned | vs. pysteps STEPS optical flow |
| **6. Streamlit Demo** | ⏳ Planned | Interactive Swiss map predictions |

