@echo off
call conda activate python_3d
python src/train/train.py --data_type=Synthetic  --flag=train --restore=0 --restore_time=2024-01-26-17-09-25