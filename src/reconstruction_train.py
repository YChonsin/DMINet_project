# Code Implementation of the training process
# you can change the parameters here or in the file 'run_train_DMINet.sh' (recommended)

import gc
import os, sys, argparse, time
import logging, datetime
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR

from dataset_processing import MyDataset

from utils import *
from torchmetrics.functional import structural_similarity_index_measure as SSIM

# argument settings and parser
parser = argparse.ArgumentParser(description='recon_training')
parser.add_argument("--batchSize", type=int, default=32,
                    help="Training batch size")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs")
parser.add_argument("--wnum", type=int, default=4,
                    help="Number of workers")
parser.add_argument("--gpunum", type=int, default=1,
                    help="Number of GPUs")
parser.add_argument("--milestone", type=int, default=15,
                    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Initial learning rate")
parser.add_argument("--net", type=str, default='DMINet128',
                    help="Selected Neural Network")
parser.add_argument("--dpath", type=str, default='recon128',
                    help="Path to the training prepared_dataset")
parser.add_argument("--thre", type=int, default=100,
                    help="Threshold to use GPU")
parser.add_argument("--dtype", type=str, default='2d',
                    help="2d: 2d data; 3d: 3d data; 5d: 5d data")
# parser.add_argument("--loss", type=str, default='MSE',
#                     help="Selected Loss Function")
parser.add_argument("--cp", type=str2bool, default=False,
                    help="Whether to use checkpoint to continue uncompleted training")
parser.add_argument("--mfolder", type=str, default='',
                    help="Folder to the checkpoint; available when [--cp] is True")
opt = parser.parse_args()

# record the time
current_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

# folder
os.makedirs(f'./trained_models/{opt.dtype}/', exist_ok=True)
save_dir = f'./trained_models/{opt.dtype}/{opt.net}{current_time}'

def main():
    # import and build network model
    from importlib import import_module
    match opt.dtype:
        case '2d':
            net_folder = "src.model."

    module_name = net_folder + opt.net
    net_module = import_module(module_name)
    net_class = getattr(net_module, opt.net)
    net = net_class()
    print("Loading model: ", opt.net)
    # use multiple GPUs for training
    device_ids = multi_gpu(opt.gpunum, opt.thre)
    cuda_visible_devices = ",".join(str(id) for id in device_ids)  # 将 device_ids 转换为逗号分隔的字符串
    # print(device_ids, cuda_visible_devices)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device(f'cuda:{device_ids[0]}')
    # load model to GPU
    model = nn.DataParallel(net, device_ids=device_ids).to(device)

    # define loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-7)
    # define the iteration of the learning rate
    milestones = []
    for i in range(opt.milestone, opt.epochs, opt.milestone):
        milestones.append(i)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # you can use checkpoint with option '--cp' if needed
    if opt.cp:
        # if opt.dtype == '2d':
        #     read_save_dir = f'./trained_models/{opt.mfolder}'
        # elif opt.dtype == '3d':
        #     read_save_dir = f'./trained_models_3d/{opt.mfolder}'
        read_save_dir = f'./trained_models/{opt.dtype}/{opt.mfolder}'
        ck_filepath = f'{read_save_dir}/checkpoint.pth'
        start_epoch, results_train, results_val, timelist = (
            load_checkpoint(model, optimizer, scheduler, ck_filepath))
        assert results_train is not None
        assert results_val is not None
        assert timelist is not None
    else:
        start_epoch = 0
        results_train = {'Loss':[], 'PSNR': [], 'SSIM': []}
        results_val = {'Loss':[], 'PSNR': [], 'SSIM': []}
        timelist = []

    # loading prepared_dataset
    print('Loading prepared_dataset ...\n')
    dpath = f'./prepared_dataset/{opt.dtype}/' + opt.dpath

    dataset_train = MyDataset(datapath=dpath, istrain=True)
    dataset_val = MyDataset(datapath=dpath, istrain=False)

    train_loader = DataLoader(dataset=dataset_train, num_workers=opt.wnum, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    print("## Training batches: %d" % int(len(train_loader)))
    val_loader = DataLoader(dataset=dataset_val, num_workers=opt.wnum, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    print("## Validating batches: %d" % (int(len(val_loader)) if val_loader is not None else 0))

    # 训练
    best = 0.
    try:
        # loop for training and validation
        for epoch in range(start_epoch, opt.epochs):
            print(f'Epoch-{epoch+1} begin.')
            print('Learning rate %f' % optimizer.param_groups[0]['lr'])

            model.train()  # 训练模式
            # define lists for metrics
            loss_list = []
            psnr_list = []
            ssim_list = []
            start_time = time.time()
            train_bar = tqdm(train_loader)
            # loop for training
            for i, (feature, label) in enumerate(train_bar):
                label, feature = label.to(device), feature.to(device)
                optimizer.zero_grad()
                predicted = model(feature)  # output
                # calculate loss and other metrics
                loss = criterion(predicted, label)
                loss.backward()
                psnr = batch_PSNR(predicted, label, torch.max(label))
                ssim = SSIM(predicted, label).item()
                optimizer.step()

                if torch.isnan(loss):
                    raise ValueError(f"Epoch {epoch}, Batch {i}: Loss is NaN, stopping training.")

                # save the metrics
                loss_list.append(loss.item())
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                train_bar.set_description(
                    desc='[%d/%d] Loss: %.10f\tPSNR: %.2f\tSSIM: %.4f' % (
                    epoch + 1, opt.epochs, loss.item(), psnr, ssim))
            # get average metrics
            epoch_loss = sum(loss_list) / len(loss_list)
            epoch_psnr = sum(psnr_list) / len(psnr_list)
            epoch_ssim = sum(ssim_list) / len(ssim_list)
            print("\n[Epoch %d] Training Avg Loss: %.10f\tAvg PSNR: %.2f\tSSIM: %.4f" %
                  (epoch + 1, epoch_loss, epoch_psnr, epoch_ssim))
            results_train['Loss'].append(epoch_loss)
            results_train['PSNR'].append(epoch_psnr)
            results_train['SSIM'].append(epoch_ssim)

            # validating
            model.eval()
            if val_loader != None:
                with torch.no_grad():
                    val_bar = tqdm(val_loader)
                    loss_list = []
                    psnr_list = []
                    ssim_list = []
                    for i, (feature, label) in enumerate(val_bar):
                        label, feature = label.to(device), feature.to(device)
                        predicted = model(feature)

                        loss = criterion(predicted, label)
                        psnr = batch_PSNR(predicted, label, torch.max(label))
                        ssim = SSIM(predicted, label).item()

                        loss_list.append(loss.item())
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        val_bar.set_description(
                            desc='[%d/%d] Loss: %.10f\tPSNR: %.2f\tSSIM: %.4f' % (
                                epoch + 1, opt.epochs, loss.item(), psnr, ssim))
                    epoch_loss = sum(loss_list) / len(loss_list)
                    epoch_psnr = sum(psnr_list) / len(psnr_list)
                    epoch_ssim = sum(ssim_list) / len(ssim_list)
                    print("\n[Epoch %d] Validating Avg Loss: %.10f\tAvg PSNR: %.2f\tSSIM: %.4f" %
                          (epoch + 1, epoch_loss, epoch_psnr, epoch_ssim))
                    results_val['Loss'].append(epoch_loss)
                    results_val['PSNR'].append(epoch_psnr)
                    results_val['SSIM'].append(epoch_ssim)
                # choose the best model with the best metrics
                metric = epoch_psnr * epoch_ssim
                if metric > best:
                    best = metric
                    if not opt.cp:
                        os.makedirs(save_dir, exist_ok=True)

                    torch.save(model.state_dict(),
                        os.path.join(save_dir if not opt.cp else read_save_dir,
                        'recon_best.pth'))
            # save the model
            if not opt.cp:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(),
                os.path.join(save_dir if not opt.cp else read_save_dir,
                f'recon{epoch + 1}.pth'))
            # save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, timelist, results_train, results_val,
                os.path.join(save_dir if not opt.cp else read_save_dir, f'checkpoint.pth'))

            scheduler.step()
            torch.cuda.empty_cache()
            end_time = time.time()
            timelist.append(end_time - start_time)
            print(f"[Epoch {epoch + 1}] Total time used: {end_time - start_time}s")

    except ValueError as e:
        print(e)
        # 释放GPU资源
        del feature, label, predicted, loss
        torch.cuda.empty_cache()
        gc.collect()
    # write training infomation into a log
    with open(os.path.join(save_dir if not opt.cp else read_save_dir,
                           'training_info.log'), 'w') as f:
        f.write("training times:\n")
        for line in timelist:
            f.write(str(line) + "\n")
        f.write("avg_train_loss:\n")
        for line in results_train["Loss"]:
            f.write(str(line) + "\n")
        f.write("avg_train_psnr:\n")
        for line in results_train["PSNR"]:
            f.write(str(line) + "\n")
        f.write("avg_train_ssim:\n")
        for line in results_train["SSIM"]:
            f.write(str(line) + "\n")
        f.write("avg_val_loss:\n")
        for line in results_val["Loss"]:
            f.write(str(line) + "\n")
        f.write("avg_val_psnr:\n")
        for line in results_val["PSNR"]:
            f.write(str(line) + "\n")
        f.write("avg_val_ssim:\n")
        for line in results_val["SSIM"]:
            f.write(str(line) + "\n")


    # draw training loss
    plt.figure(figsize=(8, 6))
    plt.plot(results_train["Loss"])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(save_dir if not opt.cp else read_save_dir,
                             'training_loss.png'))
    plt.close()

    # draw training PSNR
    plt.figure(figsize=(8, 6))
    plt.plot(results_train["PSNR"])
    plt.title('Training PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(save_dir if not opt.cp else read_save_dir,
                             'training_psnr.png'))
    plt.close()  # 关闭当前图形

# make necessary folders
os.makedirs('./logs', exist_ok=True)
os.makedirs('./results', exist_ok=True)

log_filename = f'./logs/{opt.net}_{current_time}.log'
# logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename)
                        , logging.StreamHandler()
                              ])

# create a logger
logger = logging.getLogger()
# redirect both the standard output and the standard error to the log recorder,
# while retaining the original output to the console
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''

    def write(self, message):
        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.level(line + '\n')

    def flush(self):
            if self.buffer != '':
                self.level(self.buffer.strip())
                self.buffer = ''

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

if __name__ == "__main__":
    main()



