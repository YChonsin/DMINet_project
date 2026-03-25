# Code Implementation of the training process
# you can change the parameters here or in the file 'run_train_DMINet.sh' (recommended)

import os, glob, argparse, shutil, random, time
import torch
import torch.nn as nn
import re
import ast

from skimage.metrics import structural_similarity as ssim


from utils import *
import scipy.io as sio

parser = argparse.ArgumentParser(description="Reconstruction Test")
parser.add_argument("--mtype", type=str, default='regular', help='missing type; regular: delete one out of every three; irregular: randomly delete 50% of traces')
parser.add_argument("--mrate", type=float, default='0.0', help='missing rate')
parser.add_argument("--urf", type=int, default='0', help='uniform random factor')
parser.add_argument("--folder", type=str, default='./pre_data/test', help='test data folder')
parser.add_argument("--epochs", type=int, default=0, help='epochs')
parser.add_argument("--net", type=str, default='MambaIR', help="network")
parser.add_argument("--gpunum", type=int, default=1, help="Number of GPUs")
parser.add_argument("--shape", type=str, default="(128, 128)", help="Size of patch")
parser.add_argument("--normtype", type=int, default=0, help="0: normalize before patching; 1: normalize after patching")
parser.add_argument("--mfolder", type=str, default='0000', help="**Must: the folder of the trained model")
parser.add_argument("--thre", type=int, default=100, help="")
parser.add_argument("--tmp", type=bool, default=False, help="")
parser.add_argument("--tl", type=bool, default=False, help="")
parser.add_argument("--tlfolder", type=str, default="", help="")


opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    from importlib import import_module

    module_name = "src.model." + opt.net

    net_module = import_module(module_name)
    net_class = getattr(net_module, opt.net)
    net_type = opt.net
    # 加载模型
    net = net_class()
    print("Loading model: ", opt.net)

    # 采用多GPU进行预测
    gpu_num = opt.gpunum
    device_ids = multi_gpu(gpu_num, opt.thre)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 将 device_ids 转换为逗号分隔的字符串
    cuda_visible_devices = ",".join(str(id) for id in device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(net, device_ids=device_ids)
    # model = net

    if opt.tl:
        save_dir = f'./trained_models/{opt.tlfolder}/2d/{opt.mfolder}'
    else:
        save_dir = f'./trained_models/2d/{opt.mfolder}'
    model_filename = 'recon_best.pth' if opt.epochs == 0 else f'recon{opt.epochs}.pth'
    model.load_state_dict(torch.load(os.path.join(save_dir, model_filename), map_location="cpu"))
    model = model.to(device)

    model.eval()  # 评估模式
    print('Loading data info ...\n')
    # print(os.path.join('./' + opt.folder))
    files_source = glob.glob(os.path.join(f'./data/{opt.folder}/', '*'))
    # print(files_source)
    assert len(files_source) != 0
    # files_source = glob.glob(os.path.join('./sdata/test_recon', 'field_label.mat'))
    file_to_metrics_map = {}
    reconstructed_map = {}
    arr_map = {}
    miss_arr_map = {}
    with ((torch.no_grad())):

        start_time_all = time.time()

        seed_dict = []

        for i in range(len(files_source)):
            seed_dict.append((i + 1) * 666)

        # # 针对C3_NA数据的特殊处理
        # tmp_seed = seed_dict[5]
        # seed_dict[5] = seed_dict[0]
        # seed_dict[0] = tmp_seed
        print(seed_dict) # [3996, 1332, 1998, 2664, 3330, 666, 4662, 5328]

        # 处理数据
        for f in files_source:

            block_shape = ast.literal_eval(opt.shape)

            fname = os.path.basename(f)

            # 文件名里面只能有两个数字，多了少了都不行
            combine_list = [
                'Kerry', 'parihaka',
                'C3'
            ]
            to_get_key = False
            for element in combine_list:
                if element in fname:
                    to_get_key = True
                    break
            if to_get_key:
                nums = re.findall(r'\d+', fname)
                if nums:
                    key = int(nums[1]) - 1   # 0-19
                else:
                    raise Exception("Number not found.")

            else:
                key = 0

            start_time = time.time()

            print(f"\nFile \'{fname}\' starting deleting and reconstructing\n")

            arr = load_data_auto(f, fname[:-4]) # 读取mat数据，以及其中的数组
            # 如果是合成数据“d1”，则取其上部信噪比高的部分
            # if "test_2d" in f:
            #     arr = arr[252:, :]
            n_rows, n_cols = arr.shape

            # print("size of the seismic data", arr.shape)
            # 对数据进行填充，至能整除窗口大小
            # 填充arr至可整除窗口大小
            print(arr.shape)

            arr_max = np.max(np.abs(arr))
            # print(f"Max value:{arr_max}")
            arr = arr / arr_max
            miss_arr = np.copy(arr)
            print(miss_arr.shape)
            print(f"Missing rate: {opt.mrate}")
            # 模拟不同类型的缺失
            if opt.mtype == 'regular':  # 常规缺失：每2道删除1道；
                match opt.mrate:
                    case 0.33:
                        for col in range(1, miss_arr.shape[1], 3):
                            miss_arr[:, col] = 0
                    case 0.5:
                        for col in range(1, miss_arr.shape[1], 2):
                            miss_arr[:, col] = 0
                    case 0.67:
                        for col in range(1, miss_arr.shape[1], 3):
                            miss_arr[:, col:col + 2] = 0
                    case 0.75:
                        for col in range(1, miss_arr.shape[1], 4):
                            miss_arr[:, col:col + 3] = 0
                    case _:
                        mrate = [0.33, 0.5, 0.67, 0.75]
                        raise ValueError(f"{opt.mrate} is an unsupported missing rate. \n"
                                         f"Supported rates are: {mrate}")

            elif opt.mtype == 'irregular':  # 非常规缺失：随机删除道数。为了复现，人工设置随机种子

                if opt.urf != 0:
                    miss_arr = remove_traces(arr, opt.mrate, opt.mrate,
                                             'ur', ur_factor=opt.urf, seed=seed_dict[key])
                else:
                    miss_arr = remove_traces(arr, opt.mrate, opt.mrate,
                                             'r')


            elif opt.mtype == 'complex':
                del_num = int(miss_arr.shape[1] * 0.5)
                del_cols = random.sample(range(miss_arr.shape[1]), del_num)
                for col in del_cols:
                    miss_arr[:, col] = 0
                del_num = 50
                # 左闭右开所以选择用randrange 而不是randint
                start = random.randrange(0, miss_arr.shape[1] - del_num)
                miss_arr[:, start:start + del_num] = 0
            elif opt.mtype == 'biggap':
                del_num = 10
                start = random.randrange(0, miss_arr.shape[1] - del_num)
                # print(f'start: {start}')
                miss_arr[:, start:start + del_num] = 0
            elif opt.mtype == 'none':
                pass

            arr = pad(arr, *block_shape)
            miss_arr = pad(miss_arr, *block_shape)
            print("After padding: ", arr.shape)

            if opt.normtype == 0:  # 直接输入整幅，适用于CNN-based

                with torch.no_grad():  # this can save much memory
                    miss_arr = torch.tensor(miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    reconstructed = model(miss_arr).cpu().detach().numpy().squeeze()
                miss_arr = miss_arr.cpu().detach().numpy().squeeze()

                arr = (arr * arr_max)[:n_rows, :n_cols]
                miss_arr = (miss_arr * arr_max)[:n_rows, :n_cols]
                reconstructed = (reconstructed * arr_max)[:n_rows, :n_cols]
                # arr = arr[:n_rows, :n_cols]
                # miss_arr = miss_arr[:n_rows, :n_cols]
                # reconstructed = reconstructed[:n_rows, :n_cols]

            elif opt.normtype == 1:  # (平滑处理)分片预测，然后结合
                with torch.no_grad():  # this can save much memory
                    print("Predicting...")
                    reconstructed = test_with_sliding_window_2d(
                        miss_arr, block_shape, model, device, tuple(x // 2 for x in block_shape))
                print(f"reconstructed.shape: {reconstructed.shape}")

                arr = (arr * arr_max)[:n_rows, :n_cols]
                miss_arr = (miss_arr * arr_max)[:n_rows, :n_cols]
                reconstructed = (reconstructed * arr_max)[:n_rows, :n_cols]
                arr_map.update({key: arr})   # 记录结果
                miss_arr_map.update({key: miss_arr})
                reconstructed_map.update({key: reconstructed})
                # arr = arr[:n_rows, :n_cols]
                # miss_arr = miss_arr[:n_rows, :n_cols]
                # reconstructed = reconstructed[:n_rows, :n_cols]

            elif opt.normtype == 2:  # 取其中一小片进行预测
                print(miss_arr.shape)
                miss_arr = miss_arr[0:32, :]
                # arr = arr[1600:1728, 0:128]
                print(f"After patched: {miss_arr.shape}")

                miss_arr = pad(miss_arr, *block_shape)

                miss_arr = torch.tensor(miss_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():  # this can save much memory
                    reconstructed = model(miss_arr).cpu().detach().numpy().squeeze()
                miss_arr = miss_arr.cpu().detach().numpy().squeeze()

                arr = (arr * arr_max)[0:32, :]
                miss_arr = (miss_arr * arr_max)
                reconstructed = (reconstructed * arr_max)

            else:
                raise NotImplementedError(f"Normtype {opt.normtype} not implemented.")

            # 保存为mat文件

            dtype = '2d_tmp' if opt.tmp else '2d'
            result_dir = f'./results/{dtype}/{net_type}/{os.path.basename(opt.folder)}/'

            os.makedirs(result_dir, exist_ok=True)
            sio.savemat(result_dir + f"{fname[:-4]}.mat",
                        {'complete': arr, 'missing': miss_arr,
                         'reconstructed': reconstructed, 'difference': reconstructed - arr})
            print(f"Results saved to {result_dir + f"{fname[:-4]}.mat"}")

            # 衡量数值性能
            snr_miss = array_SNR(miss_arr, arr)
            psnr_miss = array_PSNR(miss_arr, arr)
            nrmse_miss = array_NRMSE(miss_arr, arr)
            ssim_miss, _ = ssim(miss_arr, arr, full=True, data_range=arr_max)

            snr_test = array_SNR(reconstructed, arr)
            psnr_test = array_PSNR(reconstructed, arr)
            nrmse_test = array_NRMSE(reconstructed, arr)
            ssim_test, _ = ssim(reconstructed, arr, full=True, data_range=arr_max)

            end_time = time.time()

            metrics = [snr_miss, psnr_miss, nrmse_miss, ssim_miss,
                       snr_test, psnr_test, nrmse_test, ssim_test,
                       end_time - start_time]
            file_to_metrics_map.update({key: metrics})
            print(f"SNR on missing {os.path.basename(f)}: {snr_miss: .2f}")
            print(f"PSNR on missing {os.path.basename(f)}: {psnr_miss: .2f}")
            print(f"NRMSE on missing {os.path.basename(f)}:  {nrmse_miss: .4f}")
            print(f"SSIM on missing {os.path.basename(f)}: {ssim_miss: .6f}")
            print(f"SNR on reconstructed {os.path.basename(f)}: {snr_test: .2f}")
            print(f"PSNR on reconstructed {os.path.basename(f)}: {psnr_test: .2f}")
            print(f"NRMSE on reconstructed {os.path.basename(f)}:  {nrmse_test: .4f}")
            print(f"SSIM on reconstructed {os.path.basename(f)}: {ssim_test: .6f}")

            print(f"Inference time of {fname}: {end_time - start_time}s")

        end_time_all = time.time()

    with open(os.path.join(result_dir, 'metrics.log'), 'w') as result_file:
        print("----------------------------------------------------------")
        for key, value in file_to_metrics_map.items():
            print(f"SNR on missing {key}: {value[0]: .2f}")
            print(f"PSNR on missing {key}: {value[1]: .2f}")
            print(f"NRMSE on missing {key}:  {value[2]: .4f}")
            print(f"SSIM on missing {key}: {value[3]: .6f}")
            print(f"SNR on reconstructed {key}: {value[4]: .2f}")
            print(f"PSNR on reconstructed {key}: {value[5]: .2f}")
            print(f"NRMSE on reconstructed {key}:  {value[6]: .4f}")
            print(f"SSIM on reconstructed {key}: {value[7]: .6f}")
            print("")
            result_file.write(f"SNR on missing {key}: {value[0]: .2f}\n")
            result_file.write(f"PSNR on missing {key}: {value[1]: .2f}\n")
            result_file.write(f"NRMSE on missing {key}:  {value[2]: .4f}\n")
            result_file.write(f"SSIM on missing {key}: {value[3]: .6f}\n")
            result_file.write(f"SNR on reconstructed {key}: {value[4]: .2f}\n")
            result_file.write(f"PSNR on reconstructed {key}: {value[5]: .2f}\n")
            result_file.write(f"NRMSE on reconstructed {key}:  {value[6]: .4f}\n")
            result_file.write(f"SSIM on reconstructed {key}: {value[7]: .6f}\n")
            # print(f"Inference time of {key}: {value[6]}s")

        to_combine = False
        for element in combine_list:
            if element in fname:
                to_combine = True
                break

        if to_combine:
            # print(file_to_metrics_map)
            new_arr = np.zeros((arr.shape[0], arr.shape[1], len(file_to_metrics_map)))
            for k, value in arr_map.items():
                new_arr[:, :, k] = value

            new_miss_arr = np.zeros((arr.shape[0], arr.shape[1], len(file_to_metrics_map)))
            for k, value in miss_arr_map.items():
                new_miss_arr[:, :, k] = value

            print(new_miss_arr.shape)
            snr_miss = array_SNR(new_miss_arr, new_arr)
            psnr_miss = array_PSNR(new_miss_arr, new_arr)
            nrmse_miss = array_NRMSE(new_miss_arr, new_arr)
            ssim_miss, _ = ssim(new_miss_arr, new_arr, full=True, data_range=np.max(np.abs(new_arr)))

            new_reconstructed = np.zeros((arr.shape[0], arr.shape[1], len(reconstructed_map)))
            for k, value in reconstructed_map.items():
                new_reconstructed[:, :, k] = value
            snr_test = array_SNR(new_reconstructed, new_arr)
            psnr_test = array_PSNR(new_reconstructed, new_arr)
            nrmse_test = array_NRMSE(new_reconstructed, new_arr)
            ssim_test, _ = ssim(new_reconstructed, new_arr, full=True, data_range=np.max(np.abs(new_arr)))

            sio.savemat(result_dir + "combined.mat",
                        {'complete': new_arr, 'missing': new_miss_arr,
                         'reconstructed': new_reconstructed, 'difference': new_reconstructed - new_arr})
            print(f"Combined result saved to {result_dir + "combined.mat"}")

            print("--------------------combined-------------------------")
            print(f"SNR on missing combined: {snr_miss: .2f}")
            print(f"PSNR on missing combined: {psnr_miss: .2f}")
            print(f"NRMSE on missing combined:  {nrmse_miss: .4f}")
            print(f"SSIM on missing combined: {ssim_miss: .6f}")
            print(f"SNR on reconstructed combined: {snr_test: .2f}")
            print(f"PSNR on reconstructed combined: {psnr_test: .2f}")
            print(f"NRMSE on reconstructed combined:  {nrmse_test: .4f}")
            print(f"SSIM on reconstructed combined: {ssim_test: .6f}")

        print("----------------------------------------------------------")

        print(f"Processing time for all files: {end_time_all - start_time_all}s")

if __name__ == "__main__":
    main()
