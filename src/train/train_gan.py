import os
import sys
import argparse
import random
import torch
import shutil
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import Print_Logger
from dataset import DualDataset, RandomRotate, Collater
import network
import loss


def parse_argument():
    parser = argparse.ArgumentParser()
    IS_DEBUG = getattr(sys,"gettrace", None) is not None and sys.gettrace()
    if IS_DEBUG:
        parser.add_argument("--data_type", type=str, default="Synthetic", help="Data type for training [default:Research_Data]")
        parser.add_argument('--flag', type=str, default="train", help='Training flag')  
    else:
        parser.add_argument('--data_type', type=str,required=True, help='Data type for training')  
        parser.add_argument('--flag', type=str, required=True, help='Training flag')  
    # data processing
    parser.add_argument("--filter_patch_count", type=int, default=100, help="submeshes that facet count less than this will not been included in training")
    parser.add_argument("--sub_size", type=int, default=20000, help="The facet count of submesh if split big mesh[default:20000]")
    parser.add_argument('--loss_v', type=str, default='L1',
                        help='vertex loss [default: L1]')
    parser.add_argument('--loss_n', type=str, default='L1',
                        help='normal loss [default: L1]')
    parser.add_argument('--loss_v_scale', type=float,
                        default=1, help='vertex loss scale [default: 1]')
    parser.add_argument('--loss_n_scale', type=float,
                        default=1, help='normal loss scale [default: 1]')

    parser.add_argument('--wei_param', type=int, default=2)
    
    # Training
    parser.add_argument("--max_epoch", type=int, default=1000, help="The number of epochs[default:50]")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size[default:1]")
    parser.add_argument('--seed', type=int, default=None,
                        help='Manual seed [default: None]')
    parser.add_argument('--lr_sch', type=str,
                        default='lmd', help='lr scheduler')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate [default: 0.001]')
    parser.add_argument('--lr_step', type=int, nargs='+',
                        default=[10], help='Decay step for learning rate [default: 10]')
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='Decay rate for learning rate [default: 0.95]')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='adam or sgd momentum [default: adam]')   
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='First decay ratio, for Adam [default: 0.9]')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Second decay ratio, for Adam [default: 0.999]')
    parser.add_argument('--weight_decay', type=float,
                        default=0, help='Weight decay coef [default: 0]')

    # Restore
    parser.add_argument('--restore', type=int, default=0, help='')
    parser.add_argument('--restore_time', type=str, default=None, help='')
    parser.add_argument('--last_epoch', type=int, default=None, help='')

    opt, ext_list = parser.parse_known_args()
    print(f"-----------------opt----------------\n{opt}")
    print(f"-----------------ext_list----------------\n{ext_list}")
    
    # parse extra
    def to_dict(extra):
        return {extra[0]: extra[1]}
    
    for arg in ext_list:
        array_arg = arg[2:].split("=",1)
        if len(array_arg) !=2:
            continue
        opt.__dict__.update(to_dict(array_arg))
        
    # Special parameter
    opt.force_depth = True if opt.data_type in ['Kinect_v1', 'Kinect_v2'] else False
    opt.pool_type = 'max'    
    return opt

def choose_optimizer(model, opt):
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,
                              momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.9)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(
            opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    return optimizer

def choose_learningrate_schedule(optimizer, opt):
    if opt.lr_sch == 'step':
        lr_sch = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step[0], gamma=opt.lr_decay)
    elif opt.lr_sch == 'multi_step':
        lr_sch = lr_scheduler.MultiStepLR(
            optimizer, milestones=opt.lr_step, gamma=opt.lr_decay)
    elif opt.lr_sch == 'exp':
        lr_sch = lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    elif opt.lr_sch == 'auto':
        lr_sch = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=opt.lr_decay, patience=opt.lr_step[0], verbose=True)
    else:
        def lmd_lr(step):
            return opt.lr_decay**(step/opt.lr_step[0])
        lr_sch = lr_scheduler.LambdaLR(optimizer, lr_lambda=lmd_lr)
    return lr_sch

def train_one_epoch(generator, discriminator, train_dataset_loader, optimizer_generator, optimizer_discriminator, device, epoch, opt, train_writer):
    # Set up pbar
    desc ="Training - epoch: {:>3} loss {:.4f} {:.4f} {:.4f} error {:.4f} {:.4f}" #format for tqdm from internet
    bar = "{desc} ({n_fmt}/{total_fmt} | {elapsed} < {remaining})"
    pbar = tqdm(total=len(train_dataset_loader), ncols=90, leave=False, desc=desc.format(epoch, 0, 0, 0, 0, 0),bar_format=bar)
    
    # Turn on training mode
    generator.train()
    discriminator.train()
    optimizer_generator.zero_grad()    
    optimizer_discriminator.zero_grad()

    # Loop through data
    for step, data in enumerate(train_dataset_loader):
        data = [d.to(device) for d in data]

        # ==================================
        #  Discriminator 
        # ==================================
        optimizer_discriminator.zero_grad()

        # Generate fake data
        vert_predict, norm_predict, _ = generator(data)
        # vert_predict, norm_predict = vert_predict.detach(), norm_predict.detach()
        # Calculate loss
        real_vertex_logits, real_facet_logits = discriminator(data[0].y, data[1].y, data[0].edge_index, data[1].edge_index)
        fake_vertex_logits, fake_facet_logits = discriminator(vert_predict, norm_predict, data[0].edge_index, data[1].edge_index)
        discriminator_loss = loss.discriminator_loss(real_vertex_logits, real_facet_logits, fake_vertex_logits, fake_facet_logits, device)

        # Calculate gradient discriminator
        discriminator_loss /= opt.batch_size
        discriminator_loss.backward()
        discriminator_loss *= opt.batch_size

        # ==================================
        # Generator
        # ==================================
        optimizer_generator.zero_grad()
        # Generate fake data
        vert_predict, norm_predict, _ = generator(data)

        # Calculate generator Loss
        fake_vertex_logits, fake_facet_logits = discriminator(vert_predict, norm_predict, data[0].edge_index, data[1].edge_index)
        # fake_vertex_logits, fake_facet_logits = fake_vertex_logits.detach(), fake_facet_logits.detach()    
        generator_loss = loss.generator_loss(fake_vertex_logits, fake_facet_logits, device)

        # Calculate dual loss
        # ==================================
        train_loss_v = loss.loss_v(vert_predict, data[0].y, opt.loss_v)
        train_loss_f = loss.loss_n(norm_predict, data[1].y, opt.loss_n)
        dual_loss = loss.dual_loss(
                train_loss_v, train_loss_f, v_scale=opt.loss_v_scale, n_scale=opt.loss_n_scale)
        # Calculate gradient generator
        total_generator_loss = generator_loss + dual_loss
        total_generator_loss /= opt.batch_size
        total_generator_loss.backward()
        total_generator_loss *= opt.batch_size
        # Calculate error
        train_error_v = loss.error_v(vert_predict, data[0].y)
        train_error_f = loss.error_n(norm_predict, data[1].y)
        # Gradient accumulation, update when batch_size reached
        if (((step+1) % opt.batch_size) == 0) or (step+1 == len(train_dataset_loader)):
            optimizer_discriminator.step()
            optimizer_generator.step()
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            pbar.desc = desc.format(
                    epoch, train_loss_v, train_loss_f, dual_loss, train_error_v, train_error_f)
            pbar.update(opt.batch_size)
    train_writer.add_scalar("discriminator_loss", discriminator_loss.item(), epoch)
    train_writer.add_scalar("generator_loss", generator_loss.item(), epoch)
    train_writer.add_scalar('loss_v', train_loss_v.item(), epoch)
    train_writer.add_scalar('loss_f', train_loss_f.item(), epoch)
    train_writer.add_scalar('dual_loss', dual_loss.item(), epoch)
    train_writer.add_scalar('error_v', train_error_v.item(), epoch)
    train_writer.add_scalar('error_f', train_error_f.item(), epoch)
    pbar.close()

def eval_model(generator, discriminator, eval_dataset_loader, device, epoch, opt, test_writer):
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        bar = "{desc} ({n_fmt}/{total_fmt} | {elapsed} < {remaining})"
        desc = "VALIDATION - epoch:{:>3} loss:{:.4f} {:.4f}  error:{:.4f} {:.4f}"
        pbar = tqdm(total=len(eval_dataset_loader), ncols=90, leave=False,
                        desc=desc.format(epoch, 0, 0, 0, 0), bar_format=bar)
        eval_loss_discriminator = eval_loss_generator = eval_loss_v = eval_loss_f = eval_error_v = eval_error_f = count_v = count_f = 0
        num_data = len(eval_dataset_loader)
        for i, data in enumerate(eval_dataset_loader):
            data = [d.to(device) for d in data]
            vert_predict, norm_predict, _ = generator(data)
            real_vertex_logits, real_facet_logits = discriminator(data[0].y, data[1].y, data[0].edge_index, data[1].edge_index)
            fake_vertex_logits, fake_facet_logits = discriminator(vert_predict, norm_predict, data[0].edge_index, data[1].edge_index)
            loss_i_discriminator = loss.discriminator_loss(real_vertex_logits, real_facet_logits, fake_vertex_logits, fake_facet_logits, device)
            loss_i_generator = loss.generator_loss(fake_vertex_logits, fake_facet_logits, device)
            loss_i_v = loss.loss_v(vert_predict, data[0].y, opt.loss_v)
            loss_i_f = loss.loss_n(norm_predict, data[1].y, opt.loss_n)
            error_i_v = loss.error_v(vert_predict, data[0].y)
            error_i_f = loss.error_n(norm_predict, data[1].y)

            eval_loss_discriminator += loss_i_discriminator
            eval_loss_generator += loss_i_generator
            eval_loss_v += loss_i_v * data[0].num_nodes
            eval_loss_f += loss_i_f * data[1].num_nodes
            eval_error_v += error_i_v * data[0].num_nodes
            eval_error_f += error_i_f * data[1].num_nodes
            count_v += data[0].num_nodes
            count_f += data[1].num_nodes

            pbar.desc = desc.format(
                    epoch, loss_i_v, loss_i_f, error_i_v, error_i_f)
            pbar.update(1)
        pbar.close()
        eval_loss_discriminator /= num_data
        eval_loss_generator /= num_data
        eval_loss_v /= count_v
        eval_loss_f /= count_f
        eval_error_v /= count_v
        eval_error_f /= count_f
        test_writer.add_scalar('discriminator_loss', eval_loss_discriminator.item(), epoch)
        test_writer.add_scalar('generator_loss', eval_loss_generator.item(), epoch)
        test_writer.add_scalar('loss_v', eval_loss_v.item(), epoch)
        test_writer.add_scalar('loss_f', eval_loss_f.item(), epoch)
        test_writer.add_scalar('error_v', eval_error_v.item(), epoch)
        test_writer.add_scalar('error_f', eval_error_f.item(), epoch)
    return eval_loss_v, eval_loss_f, eval_error_v, eval_error_f
def train(opt):

    
    # Prepare the training information
    print("\n--------------------Training start------------------------")
    training_name = f"GeoBi-GNN_{opt.data_type}"
    if opt.restore:
        training_time = opt.restore_time
    else:
        training_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    flag = f"{training_name}_{opt.flag}"
    print("Training_flag: ", flag)

    # Seed
    if opt.seed is None:
        opt.seed = random.randint(1,10000)
    # opt.seed = 42
    print(f"Random Seed: {opt.seed}\n")
    # random.seed(opt.seed)
    # np.random.seed(opt.seed)
    # torch.manual_seed(opt.seed)
    # torch.cuda.manual_seed(opt.seed)
    # torch.backends.cudnn.deterministic = True
    
    # 1. Prepare path
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(CODE_DIR,"log")
    log_dir = os.path.join(LOG_DIR, flag, training_time)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Print_Logger(os.path.join(log_dir, "training_info.txt"))
    opt.model_name = f"{training_name}_model.pth"
    opt.params_name = f"{training_name}_params.pth"
    opt.restore_name = f"{training_name}_restore.pth"
    model_path = os.path.join(log_dir, opt.model_name)
    params_path = os.path.join(log_dir, opt.params_name)
    restore_path = os.path.join(log_dir, opt.restore_name)
    torch.save(opt, params_path)
    # Restore
    if opt.restore:
        restore_params = torch.load(restore_path)
        restore_last_epoch = restore_params.get('last_epoch', None)
        restore_best_error = restore_params.get('best_error', None)
    else:
        restore_params = dict()

    # tensorboard
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    test_writer.add_text('train_params', str(opt))
    
    # 2. Prepare data
    print("==="*30)
    train_dataset = DualDataset(opt.data_type, train_or_test="train", filter_patch_count=opt.filter_patch_count, submesh_size=opt.sub_size, transform = RandomRotate(False))
    train_dataset_loader = DataLoader(train_dataset, shuffle=True, collate_fn=Collater([]))
    eval_dataset = DualDataset(opt.data_type, train_or_test='test', submesh_size=opt.sub_size)
    print(f"train_dataset:{len(train_dataset):>4} samples")
    print(f"Testing set:{len(eval_dataset):>4} samples")
    
    # 3. Prepare Model
    generator = network.DualGenerator(force_depth=opt.force_depth, pool_type=opt.pool_type, wei_param=opt.wei_param)
    discriminator = network.DualDiscriminator(opt.force_depth, pool_type=opt.pool_type, wei_param=opt.wei_param)
    total_params_generator = sum(p.numel() for p in generator.parameters())
    total_params_discriminator = sum(p.numel() for p in discriminator.parameters())
    print("Total parameters generator: ", total_params_generator)
    print("Total parameters discriminator: ", total_params_discriminator)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.restore:
        last_epoch = restore_last_epoch + 1
        best_error = restore_best_error
        generator.load_state_dict(restore_params.get("generator"))
        discriminator.load_state_dict(restore_params.get("discriminator"))
    else:
        best_error = float("inf")
        last_epoch = 0
    print(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizer_generator = choose_optimizer(generator, opt)
    optimizer_discriminator = choose_optimizer(discriminator, opt)

    lr_sch_optimizer = choose_learningrate_schedule(optimizer_generator, opt)
    lr_sch_discriminator = choose_learningrate_schedule(optimizer_discriminator, opt) 
        
    # 4. Training
    time_start = datetime.now()

    for epoch in range(last_epoch, opt.max_epoch):

        print_log = epoch%10==0
        train_one_epoch(generator, discriminator, train_dataset_loader, optimizer_generator, optimizer_discriminator, device, epoch, opt, train_writer)
        eval_loss_v, eval_loss_f, eval_error_v, eval_error_f = eval_model(generator, discriminator, eval_dataset, device, epoch, opt, test_writer)
        if opt.lr_sch == 'auto':
            lr_sch_optimizer.step(eval_error_f)
            lr_sch_discriminator.step(eval_error_f)
        else:
            lr_sch_optimizer.step()
            lr_sch_discriminator.step()

        # Log info
        last_lr = optimizer_generator.param_groups[0]['lr']
        span = datetime.now() - time_start
        str_log = F"Epoch {epoch:>3}: {str(span).split('.')[0]:>8}  loss:{eval_loss_v:.4f} {eval_loss_f:.4f} | "
        str_log += F"error:{eval_error_v:.4f} {eval_error_f:.4f}  lr:{last_lr:.4e}"

        # save better model
        if eval_error_f < best_error:
            # Save model
            best_error = eval_error_f
            torch.save(generator.state_dict(), model_path)
            str_log = str_log + " - save model"
            print_log = True
            # save info
            restore_params["last_epoch"] = epoch
            restore_params["best_error"] = best_error
            restore_params["generator"] = generator.state_dict()
            restore_params["discriminator"] = discriminator.state_dict()
            torch.save(restore_params, restore_path)

        if print_log:
            tqdm.write(str_log)

    train_writer.close()
    test_writer.close()
    print(F"\n{flag}\nbest error: {best_error}")
    print('==='*30)
    print("\n--------------------Training end--------------------------")
    return os.path.join(log_dir, params_path)           
            


def main():
    opt = parse_argument()
    params_file = train(opt)
    # params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"log", "GeoBi-GNN_Synthetic_2024-01-09-22-11-14", "GeoBi-GNN_Synthetic_params.pth")
    
    from test_result import predict_dir
    predict_dir(params_file, data_dir=None, sub_size=opt.sub_size, gpu=-1)

if __name__ == "__main__":
    main()
