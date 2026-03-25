# useful functions
import torch
import numpy as np
import torch.utils.data as udata
from utils import *
import glob, math, shutil, h5py
import scipy.io as sio


# The features and labels of the training set are respectively placed in train/feature and train/label,
# while the validation set is stored as val/feature and train/label.
# **Each data patch is saved as an independent .npy file.

def prepare_data_seis_for_random_recon(data_path, output_path, patch_size, stride: tuple,
                                        remove_zero = False, misstype='random',
                                       train_num=10000, val_num=1000):
    '''
        this function patches the data (.mat format) and saves the patches into .npy format
    '''
    size0, size1 = patch_size
    stride0, stride1 = stride

    shutil.rmtree(output_path, ignore_errors=True)
    print("Input path: ", data_path)
    full_dataset = glob.glob(os.path.join(data_path, '*.mat'))

    for path in [f'{output_path}/train/feature', f'{output_path}/train/label',
                 f'{output_path}/val/feature', f'{output_path}/val/label',]:
        os.makedirs(path, exist_ok=True)
    files = glob.glob(f'{output_path}/test/*')
    for file in files:
        os.remove(file)

    patches = []
    for file_path in full_dataset:
        mat_data = sio.loadmat(file_path)
        arr = mat_data[os.path.basename(file_path)[:-4]]
        arr = arr / np.max(np.abs(arr))
        tmp_patches = Ar2Patch(arr, size0, size1,
                               stride0, stride1)
        # remove patches who contain a large proportion of zeros
        if remove_zero:
            for patch in tmp_patches:
                # 去除全零patch
                nonzero_ratio = np.count_nonzero(patch) / float(patch.size)
                if nonzero_ratio >= 0.4:
                    patches += [patch]
        else:
            for patch in tmp_patches:
                patches += [patch]
    print(f'Total patches: {len(patches)}')
    assert len(patches) >= train_num + val_num

    patches = random.sample(patches, train_num + val_num)

    # get the training patches and validating patches
    train_patches, val_patches = torch.utils.data.random_split(patches, [train_num, val_num])

    count = 0
    print('Processing training data')
    for patch in train_patches:
        match misstype:
            case 'random':
                miss_patch = remove_traces(patch, 0.4, 0.6,
                                              'r', 0.)
            case 'biggap':
                miss_patch = remove_traces(patch, 0.1, 0.3,
                                              'c', 0.)

        np.save(f'{output_path}/train/label/{count}.npy', patch)
        np.save(f'{output_path}/train/feature/{count}.npy', miss_patch)
        count += 1


    print('\nProcessing validation data\n')
    count_v = 0
    for patch in val_patches:
        match misstype:
            case 'random':
                miss_patch = remove_traces(patch, 0.5, 0.7,
                                              'r', 0.)
            case 'biggap':
                miss_patch = remove_traces(patch, 0.1, 0.3,
                                              'c', 0.)

        np.save(f'{output_path}/val/label/{count_v}.npy', patch)
        np.save(f'{output_path}/val/feature/{count_v}.npy', miss_patch)
        count_v += 1

    # 输出数据集信息
    print('\nTraining set: %d patches' % count)
    print('Validation set: %d patches' % count_v)

class MyDataset(udata.Dataset):
    '''
        used to load the dataset from .npy to tensor
    '''
    def __init__(self, istrain, datapath='./data'):
        super(MyDataset, self).__init__()
        self.label_files = glob.glob(f'{datapath}/train/label/*.npy' if istrain else f'{datapath}/val/label/*.npy')
        self.feature_files = glob.glob(f'{datapath}/train/feature/*.npy' if istrain else f'{datapath}/val/feature/*.npy')
        self.label_files = sorted(self.label_files)
        self.feature_files = sorted(self.feature_files)

        # print(len(self.label_files))
        # print(len(self.feature_files))

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        if (os.path.splitext(os.path.basename(self.label_files[index]))[0] !=
                os.path.splitext(os.path.basename(self.feature_files[index]))[0]):
            raise ValueError('Label and feature files do not match')
        label = np.load(self.label_files[index])
        feature = np.load(self.feature_files[index])

        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        # print(type(feature))
        return feature, label


if __name__ == '__main__':
    patch_size = (128, 128)
    stride = (40, 40)
    train_num = 11264   # this setting accords with the manuscript
    val_num = 1408
    input_path = './data/for_training'
    output_path = './prepared_dataset/2d/recon128'
    prepare_data_seis_for_random_recon(input_path, output_path,
                                        patch_size, stride,
                                        remove_zero=True, misstype='random',
                                       train_num=train_num, val_num=val_num)

