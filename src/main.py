import sys
import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
sys.path.insert(0, path)
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
from stpyvista.utils import is_the_app_embedded 
import torch
import openmesh as om
from test_result import inference
from network import DualGenerator

# Initial configuration
pv.start_xvfb()
st.session_state.is_app_embedded = st.session_state.get(
    "is_app_embedded", is_the_app_embedded()
)
def render_mesh(mesh_file, model):
   if mesh_file is not None:

        ## Initial progress bar and column
        progress_bar = st.progress(0)
        col1, col2 = st.columns(2)

        ## Initialize plotter
        plotter_noisy = pv.Plotter(border=True, window_size=[400, 400])
        plotter_noisy.background_color = "#f0f8ff"
        plotter_denoised = pv.Plotter(border=True, window_size=[400, 400])
        plotter_denoised.background_color = "#f0f8ff"

        ## read mesh from file 
        noisy_mesh_file_temp = "./temp.obj"
        with open(noisy_mesh_file_temp, "wb") as f: 
            f.write(mesh_file.getbuffer())
        noisy_mesh = pv.read(noisy_mesh_file_temp)

        ## denoise mesh
        denoised_mesh_file_temp = "./denoised_temp.obj"
        denoised_mesh,_ = inference(noisy_mesh_file_temp, model, sub_size=1000)
        om.write_mesh(denoised_mesh_file_temp, denoised_mesh)
        denoised_mesh = pv.read(denoised_mesh_file_temp)

        ## clear temp file
        os.remove(noisy_mesh_file_temp)
        os.remove(denoised_mesh_file_temp)

        ## Update progress bar
        progress_bar.progress(100)

        ## Add mesh to the plotter
        with col1:
            plotter_noisy.add_mesh(noisy_mesh, color="orange")
            plotter_noisy.view_isometric()
            stpyvista(plotter_noisy, key=f"{mesh_file.name}_noisy")
        with col2:
            plotter_denoised.add_mesh(denoised_mesh, color="orange")
            plotter_denoised.view_isometric()
            stpyvista(plotter_denoised, key=f"{mesh_file.name}_denoised")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weight_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    model_name = "GeoBi-GNN_Synthetic_model.pth"
    model = DualGenerator(force_depth=False,
                          pool_type="max", wei_param=2)
    model.load_state_dict(torch.load(os.path.join(model_weight_folder, model_name), map_location=device))
    model = model.to(device)
    return model

def main():
    from stpyvista.utils import start_xvfb

    if "IS_XVFB_RUNNING" not in st.session_state:
        start_xvfb()
        st.session_state.IS_XVFB_RUNNING = True 

    st.title("Remove Noise")

    # Upload mesh file
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    mesh_files = st.sidebar.file_uploader("Upload your mesh",accept_multiple_files=True, type=["obj"], key=st.session_state["file_uploader_key"])

    # Clear button
    if st.sidebar.button('Reset'):
        st.session_state["file_uploader_key"] += 1
        st.rerun()

    # Prepare model
    model = load_model()

    if mesh_files is not None:
        for mesh_file in mesh_files:
            render_mesh(mesh_file, model)

 


if __name__ == "__main__":
    main()