@echo off
call conda activate python_3d
python src/train/train_gan.py --data_type=Synthetic --flag=train --restore=0 --restore_time=2024-02-05-19-17-20