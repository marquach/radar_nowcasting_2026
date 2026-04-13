import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

from pilot.model.u_net import UNet
from pilot.create_dataset import RainPrecipitationMCH
import pysteps.visualization as pyvis

st.set_page_config(page_title="Precipitation Nowcasting", layout="wide")
st.title("Precipitation Nowcasting — U-Net Demo")


@st.cache_resource
def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(7, 1)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


@st.cache_resource
def load_dataset(data_path):
    return RainPrecipitationMCH(data_path, window_size=7, target_size=1)


# --- Sidebar ---
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model checkpoint path", value="pilot/model/unet.pth")
data_path = st.sidebar.text_input("Data directory", value="pilot/data")

# Load data
ds = load_dataset(data_path)
n_samples = len(ds)

st.sidebar.markdown(f"**Dataset size:** {n_samples} samples")

# Sample selection
mode = st.sidebar.radio("Sample selection", ["Random", "Manual"])
if mode == "Random":
    n_show = st.sidebar.slider("Number of samples", 1, 6, 3)
    if st.sidebar.button("Resample"):
        st.session_state["indices"] = random.sample(range(n_samples), n_show)
    if "indices" not in st.session_state:
        st.session_state["indices"] = random.sample(range(n_samples), n_show)
    indices = st.session_state["indices"]
else:
    idx = st.sidebar.number_input("Sample index", 0, n_samples - 1, 0)
    indices = [idx]

# Load model
try:
    model = load_model(model_path)
    model_loaded = True
except Exception as e:
    st.sidebar.warning(f"Could not load model: {e}")
    model_loaded = False

# --- Main area ---
orig_h, orig_w = ds.data.shape[1], ds.data.shape[2]

for sample_idx in indices:
    st.markdown(f"### Sample {sample_idx}")
    x, y = ds[sample_idx]

    # Ground truth
    y_up = F.interpolate(y.unsqueeze(0), (orig_h, orig_w), mode="bilinear").squeeze()
    y_mm = ds.inverse_transform(y_up)

    col1, col2 = st.columns(2)

    with col1:
        fig_gt, ax_gt = plt.subplots(figsize=(6, 5))
        plt.sca(ax_gt)
        pyvis.plot_precip_field(y_mm, axis="on")
        ax_gt.set_title("Ground Truth")
        st.pyplot(fig_gt)
        plt.close(fig_gt)

    # Prediction
    if model_loaded:
        with torch.no_grad():
            pred = model(x.unsqueeze(0))
        pred_up = F.interpolate(pred, (orig_h, orig_w), mode="bilinear").squeeze()
        pred_mm = ds.inverse_transform(pred_up)

        with col2:
            fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
            plt.sca(ax_pr)
            pyvis.plot_precip_field(pred_mm, axis="on")
            ax_pr.set_title("Prediction")
            st.pyplot(fig_pr)
            plt.close(fig_pr)

        # Input sequence expander
        with st.expander(f"Input sequence (7 frames) — Sample {sample_idx}"):
            cols = st.columns(7)
            for frame_i in range(7):
                frame = x[frame_i].unsqueeze(0).unsqueeze(0)
                frame_up = F.interpolate(frame, (orig_h, orig_w), mode="bilinear").squeeze()
                frame_mm = ds.inverse_transform(frame_up)
                fig_f, ax_f = plt.subplots(figsize=(3, 2.5))
                plt.sca(ax_f)
                pyvis.plot_precip_field(frame_mm, axis="on")
                ax_f.set_title(f"t-{(7 - frame_i) * 5} min")
                with cols[frame_i]:
                    st.pyplot(fig_f)
                plt.close(fig_f)
    else:
        with col2:
            st.info("Load a model checkpoint to see predictions.")

    st.divider()
