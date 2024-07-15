# Project Name: 

## Description

This project is code base for project 3D Reconstruction that denoise noise from mesh

## Installation

### Prerequisites

Before installing, make sure you have the following installed:

- Python (version >= 3.8)
- Pip
- Virtualenv (optional but recommended)

### Installing

1. Clone the repository:
```bash
git clone https://github.com/lam1101999/3D-Reconstruction.git
```
2. Install the project dependencies
```bash
pip install -r requirements.txt
```
3. Train project
3.1 Prepare dataset
In dataset folder copy your datset with this format
```
dataset
|--yourdataset
   |--train
      |--noisy
         sameple_A_n0.obj
         sameple_A_n1.obj
         sameple_B_n.obj
      |--origin
         samepleA.obj
         sampleB.obj
   |--test   
      |--noisy
         sampleC_n.obj
      |--origin
         sampleC.obj
```
In window edit file train.bat. This script will run train_gan.py with some args
--data-type: your dataset
--restore: 0 if start new train 1 if you want to continue the previous train
--restore_time: time or name of previous train. It is saved in src/log
Ex:
```bash
python src/train/train_gan.py --data_type=Synthetic --flag=train --restore=0 --restore_time=2024-02-05-19-17-20
```

4. Web Application
set directory to src, and use streamlit to run main file
```bash
cd src
streamlit run  main.py
```
