@echo off
call conda activate python_3d
python src/train/train.py --data_type=Synthetic --max_epoch=100 --flag=train --restore=1 --restore_time=2024-01-12-05-26-11 --last_epoch=5