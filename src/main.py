import sys
import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
sys.path.insert(0, path)
import platform
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
import torch
import openmesh as om
import tempfile
import numpy as np


from test_result import inference_from_mesh
from network import DualGenerator

if platform.system() == "Linux":
    pv.start_xvfb()
    from stpyvista.utils import is_the_app_embedded 
    st.session_state.is_app_embedded = st.session_state.get(
        "is_app_embedded", is_the_app_embedded()
    )
    from stpyvista.utils import start_xvfb
    if "IS_XVFB_RUNNING" not in st.session_state:
        start_xvfb()
        st.session_state.IS_XVFB_RUNNING = True 

def convert_mesh_from_openmesh2pyvista(openmesh_mesh):
    fv_indices = openmesh_mesh.fv_indices()
    temp_array = np.full((fv_indices.shape[0],1),3)
    pyvista_mesh = pv.PolyData(openmesh_mesh.points().astype(float), np.concatenate((temp_array, fv_indices),1))
    return pyvista_mesh

def convert_mesh_from_pyvista2openmesh(pyvista_mesh):
    points = pyvista_mesh.points
    faces = np.reshape(pyvista_mesh.faces,(-1,4))[:,1:4]
    openmesh_mesh = om.TriMesh(points, faces)
    return openmesh_mesh

def render_mesh(mesh_file, model, device="cpu"):
   if mesh_file is not None:

        ## Initial progress bar and column
        progress_bar = st.progress(0)
        col1, col2 = st.columns(2)

        ## Initialize plotter
        plotter_noisy = pv.Plotter(border=True, window_size=[400, 400])
        plotter_noisy.background_color = "#f0f8ff"
        plotter_denoised = pv.Plotter(border=True, window_size=[400, 400])
        plotter_denoised.background_color = "#f0f8ff"
        progress_bar.progress(20)

        ## read mesh from file 
        noisy_mesh_file_temp = "./temp.obj"
        f = open(noisy_mesh_file_temp, "wb")
        try:
            f.write(mesh_file.getbuffer())
            reader = pv.OBJReader(f.name)
            noisy_mesh = reader.read()
            progress_bar.progress(50)
            ## denoise mesh
            denoised_mesh,_ = inference_from_mesh(convert_mesh_from_pyvista2openmesh(noisy_mesh), model, device,sub_size=20000)
            fv_indices = denoised_mesh.fv_indices()
            temp_array = np.full((fv_indices.shape[0],1),3)
            denoised_mesh = pv.PolyData(denoised_mesh.points().astype(float), np.concatenate((temp_array, fv_indices),1))
            progress_bar.progress(90)
        finally:
            f.close()
            os.remove(noisy_mesh_file_temp)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()

    if mesh_files is not None:
        for mesh_file in mesh_files:
            render_mesh(mesh_file, model, device)

 


if __name__ == "__main__":
    main()