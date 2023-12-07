import os
import sys
import argparse
import random
import torch
import shutil

import numpy as np

from tqdm import tqdm
from datetime import datetime
from utils import Print_Logger
from torch.utils.tensorboard import SummaryWriter
from dataset import DualDataset


def parse_argument():
    parser = argparse.ArgumentParser()
    IS_DEBUG = getattr(sys,"gettrace", None) is not None and sys.gettrace()
    if IS_DEBUG:
        parser.add_argument("--data_type", type=str, default="Research_Data", help="Data type for training [default:Research_Data]")
    else:
        parser.add_argument('--data_type', type=str,required=True, help='Data type for training')    
    # data processing
    parser.add_argument("--filter_patch_count", type=int, default=100, help="submeshes that facet count less than this will not been included in training")
    parser.add_argument("--sub_size", type=int, default=20000, help="The facet count of submesh if split big mesh[default:20000]")
    parser.add_argument("--epochs", type=int, default=50, help="The number of epochs[default:50]")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size[default:32]")
    parser.add_argument("--restore", type=bool, default=False, help="Whether to restore model[default:False]")
    parser.add_argument("--model_path", type=str, default=None, help="The path of model[default:None]")
    parser.add_argument('--seed', type=int, default=None,
                        help='Manual seed [default: None]')
    
    opt, ext_list = parser.parse_known_args()
    print(f"-----------------opt----------------\n{opt}")
    print(f"-----------------ext_list----------------\n{ext_list}")
    def to_dict(extra):
        return {extra[0]: extra[1]}
    
    for arg in ext_list:
        array_arg = arg[2:].split("=",1)
        if len(array_arg) !=2:
            continue
        opt.__dict__.update(to_dict(array_arg))
    
    return opt
def train(opt):

    
    # Prepare the training information
    print("\n--------------------Training start------------------------")
    training_name = f"GeoBi-GNN_{opt.data_type}"
    training_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    flag = f"{training_name}_{training_time}"
    print("Training_flag: ", flag)
    
    # Seed
    if opt.seed is None:
        opt.seed = random.randint(1,10000)
    print(f"Random Seed: {opt.seed}\n")
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    # Prepare path
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(CODE_DIR,"log")
    log_dir = os.path.join(LOG_DIR, flag)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Print_Logger(os.path.join(log_dir, "training_info.txt"))
    opt.model_name = f"{training_name}_model.pth"
    opt.params_name = f"{training_name}_params.pth"
    model_path = os.path.join(log_dir, opt.model_name)
    params_path = os.path.join(log_dir, opt.params_name)
    torch.save(opt, params_path)
    print(str(opt))

    # tensorboard
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    test_writer.add_text('train_params', str(opt))
    
    # Prepare data
    print("==="*30)
    train_dataset = DualDataset(opt.data_type, train_or_test="train", filter_patch_count=opt.filter_patch_count, submesh_size=opt.sub_size)
    
    print("\n--------------------Training end--------------------------")
def main():
    opt = parse_argument()
    params_file = train(opt)

if __name__ == "__main__":
    main()