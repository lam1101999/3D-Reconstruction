@echo off
call conda activate python_3d
python src/train/test_result.py --params_path="E:\thesis\3D-Reconstruction\src\train\log\GeoBi-GNN_Synthetic_2024-01-09-23-08-23\GeoBi-GNN_Synthetic_params.pth" --gpu=0  --sub_size=20000