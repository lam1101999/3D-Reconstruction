@echo off
call conda activate python_3d
python src/train/test_result.py --params_path="D:\project\3D-Reconstruction\src\train\log\GeoBi-GNN_Synthetic_train\2024-01-17-14-33-31\GeoBi-GNN_Synthetic_params.pth" --gpu=0  --sub_size=20000