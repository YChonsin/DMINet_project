import argparse
import subprocess
import torch
import numpy as np
import random, os
import h5py
import scipy.io as sio

def multi_gpu(gpu_num, thre=100):
    '''

    :param gpu_num: 所需GPU的数量
    :return:  selected_gpus: 可用GPU的ID列表，如果为空列表则无可用
    '''
    selected_gpus = []
    if torch.cuda.is_available():
        # 获取可用的GPU数量
        available_gpus = torch.cuda.device_count()

        used_gpus = []
        # 使用nvidia-smi获取正在使用的 GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE)
        gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
        gpu_memory = [int(x) for x in gpu_memory]

        # 如果该GPU的显存占用大于100M，则认为被使用了
        for i in range(available_gpus):
            if gpu_memory[i] > thre:
                used_gpus.append(i)

        free_gpus = [i for i in range(available_gpus) if i not in used_gpus]

        if gpu_num > len(free_gpus):
            print(f"Requested {gpu_num} GPUs, but only {len(free_gpus)} are available.")
            return []

        print(f"Number of available GPUs: {len(free_gpus)}")
        # 选择空闲的 GPU
        selected_gpus = free_gpus[:gpu_num]
        print(f"Available GPUs: {selected_gpus}")

    if len(selected_gpus) == 0:
        raise ValueError("No GPU devices available")
    return selected_gpus

def batch_PSNR(predict, target, max_value=1.):
    mse = torch.mean((predict - target) ** 2)
    psnr = 10 * torch.log10((max_value ** 2) / mse)
    return psnr.item()


def array_PSNR(prediction, target):
    mse = np.mean((target - prediction) ** 2)

    if mse == 0:
        return float('inf')

    MAX = np.max(target)

    psnr = 10 * np.log10((MAX ** 2) / mse)

    return psnr

def array_SNR(prediction, target):
    """
    计算处理后数据的信噪比（SNR）。

    :param target: 2D numpy array，表示原始信号数据
    :param prediction: 2D numpy array，表示处理后的数据
    :return: 信噪比，以分贝（dB）为单位
    """
    # 计算信号的能量
    signal_energy = np.sum(np.square(target))

    # 计算噪声（处理后数据与原始数据之差）的能量
    noise = prediction - target
    noise_energy = np.sum(np.square(noise))

    # 计算信噪比 (SNR)
    snr = 10 * np.log10(signal_energy / noise_energy)

    return snr


def array_RMSE(predict, target):
    """
    计算均方根误差（RMSE）。

    :param target: 2D numpy array，表示原始信号数据
    :param predict: 2D numpy array，表示处理后的数据或预测数据
    :return: 均方根误差（RMSE）
    """
    # 计算误差（处理后数据与原始数据之差）
    error = predict - target

    # 计算均方根误差
    mse = np.mean(np.square(error))
    rmse = np.sqrt(mse)

    return rmse

def array_NRMSE(predict, target):
    return array_RMSE(predict, target) / (np.max(target) - np.min(target))




def Ar2Patch(arr, win0, win1, stride0, stride1, padding=True):
    '''
    Cut an 2D-array into patches
    :param arr: 2D-array
    :param win1:
    :param win0:
    :param stride1:
    :param stride0:
    :return: network3D-array, the 1st dimension acts as a list
    '''
    rows, cols = arr.shape
    # print("shape of arr: ", arr.shape)
    # print("stride: ", stride_x, stride_y)
    # 计算需要添加的填充值
    if padding:
        pad0 = (stride0 - (rows - win0) % stride0) % stride0
        pad1 = (stride1 - (cols - win1) % stride1) % stride1
        # print("to be padded", pad_y, pad_x)
        # 添加填充
        arr_padded = np.pad(arr, ((0, pad0), (0, pad1)), mode='constant', constant_values=0)
        # print("padded shape: ", arr_padded.shape)
        # 提取小patch
        patches = []
        padded_rows, padded_cols = arr_padded.shape
        for row in range(0, padded_rows - win0 + 1, stride0):
            for col in range(0, padded_cols - win1 + 1, stride1):
                patch = arr_padded[row:row + win0, col:col + win1]
                patches.append(patch)
        return np.array(patches)
    else:
        patches = []
        for row in range(0, rows - win0 + 1, stride0):
            for col in range(0, cols - win1 + 1, stride1):
                patch = arr[row:row + win0, col:col + win1]
                patches.append(patch)
        return np.array(patches)

def remove_traces(arr, low=0.1, high=0.3, misstype='r',
                  zero_value = 0., ur_factor=4, seed=-1):
    '''
    第一个维度：时间采样点  第二个维度：道数
    :param arr: 2d numpy array
    :param low:
    :param high:
    :param misstype: 'r' for random, 'p' for pattern, 'c' for consecutive
    :return:
    '''
    arr = np.copy(arr)
    rows, cols = arr.shape
    if low > high:
        raise ValueError('low must be less than high')
    if low >= 0. and high < 1.:
        del_num = int(random.uniform(low, high) * cols)
    elif low == high:
        del_num = int(low * cols)
    elif low >= 1:
        del_num = random.randrange(low, high)

    match misstype:
        case 'r':
            del_cols = random.sample(range(cols), del_num)
            for col in del_cols:
                arr[:, col] = zero_value

        case 'p':
            step = cols // del_num
            for i in range(del_num):
                arr[:, i * step] = zero_value

        case 'c':
            start_col = random.randrange(0, cols - del_num)
            arr[:, start_col:start_col + del_num] = zero_value

        case 'ur':  # randomly deleting in a uniform way
            if seed != -1:
                random.seed(seed)
                print(f"Using seed {seed}")
            for i in range(0, cols, ur_factor):
                subset = ur_factor if cols - i > ur_factor else cols - i
                sub_del_num = int(random.uniform(low, high) * subset)
                sub_del_cols = random.sample(range(subset), sub_del_num)
                # print(subset, sub_del_num, sub_del_cols)

                for col in sub_del_cols:
                    arr[:, i + col] = zero_value
        case _:
            raise NotImplementedError('Unknown misstype')

    return arr

def str2bool(v):
    '''
    Used when you input **bool** parameters in command line,
    in case the parser can't recognize bool values.
    :param v:
    :return:
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def pad(arr, d0, d1):
    num_rows, num_cols = arr.shape
    pad_rows = (d0 - (num_rows % d0)) % d0
    pad_cols = (d1 - (num_cols % d1)) % d1

    # Perform the padding
    padded_arr = np.pad(arr, ((0, pad_rows), (0, pad_cols))
                        , mode='constant', constant_values=0)

    return padded_arr

def test_with_sliding_window_2d(arr,
                                patch_shape: tuple,
                                model,
                                device,
                                stride: tuple):

    step0, step1 = stride

    # 获取三维数组的大小 (depth, height, width)
    shape0, shape1 = arr.shape

    # 初始化重建后的数组和计数矩阵
    reconstructed = np.zeros_like(arr)
    count_matrix = np.zeros_like(arr)  # 用来记录重叠区域的加权

    count = 0
    # 三维滑动窗口分块
    for d in range(0, shape0 - patch_shape[0] + 1, step0):
        for i in range(0, shape1 - patch_shape[1] + 1, step1):
                # 提取当前的三维 patch
                patch = arr[d:d + patch_shape[0], i:i + patch_shape[1]]

                print(f'predicting {count}-th patch, '
                      f'[{d}:{d + patch_shape[0]}, {i}:{i + patch_shape[1]}]')
                count += 1
                # 转换为 PyTorch tensor，并加上 batch 和 channel 维度
                patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # 进行预测，不计算梯度
                with torch.no_grad():
                    reconstructed_patch = model(patch_t).cpu().numpy().squeeze()

                # 将重建后的 patch 叠加到对应的位置
                reconstructed[d:d + patch_shape[0], i:i + patch_shape[1]] += reconstructed_patch
                count_matrix[d:d + patch_shape[0], i:i + patch_shape[1]] += 1

    # 处理重叠区域，取平均值
    reconstructed /= np.maximum(count_matrix, 1)

    return reconstructed

def load_data_auto(filepath, key=None):
    """
    自动识别 .mat 文件格式（v7.3 或旧版）并读取内容。
    :param filepath: .mat 文件路径
    :return: 读取后的数据字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if filepath.lower().endswith('.mat'):
        try:
            # 尝试用 scipy 加载（适用于 v7.2 及以下）
            data = sio.loadmat(filepath)
            print(f"使用 scipy.io 读取成功（旧版 MAT 文件）: {filepath}")
            return data[key]
        except NotImplementedError as e:
            print(f"scipy.io 无法读取，可能是 v7.3 HDF5 文件，尝试使用 h5py：{e}")
        except Exception as e:
            print(f"scipy.io.loadmat 其他错误：{e}")
            raise e

    if filepath.lower().endswith('.mat') or \
        filepath.lower().endswith('.h5'):
        # 如果是 v7.3 格式，使用 h5py 读取
        try:
            with h5py.File(filepath, 'r') as f:
                print(f"使用 h5py 成功打开 HDF5 格式的 MAT 文件：{filepath}")
                key = '/' + key
                data = f[key][:]
                data = data.transpose()
                return data
        except Exception as e:
            print(f"h5py 读取失败：{e}")
            raise e

    if filepath.lower().endswith('.npy'):
        data = np.load(filepath)
        return data

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

def save_checkpoint(model, optimizer, scheduler, epoch,
                    timelist, results_train, results_val, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'results_train': results_train,
        'results_val': results_val,
        'timelist': timelist,
    }, file_path)

def load_checkpoint(model, optimizer, scheduler,
                     file_path = ""):
    checkpoint = torch.load(file_path, map_location="cpu")
    # checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    results_train = checkpoint['results_train']
    results_val = checkpoint['results_val']
    timelist = checkpoint['timelist']
    start_epoch = checkpoint['epoch'] + 1  # 从保存的下一个epoch开始
    return start_epoch, results_train, results_val, timelist


if __name__ == '__main__':
    pass