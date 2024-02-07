@echo off
call conda activate python_3d
python src/train_v2/train_gan.py --data_type=Synthetic  --flag=train --restore=0 --restore_time=2024-02-04-08-36-29