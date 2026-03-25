import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse


def plot_and_save(img, name, dt):
    '''
        plot and save seismic data
    '''
    n_time, n_trace = img.shape

    # 构造坐标轴
    time = np.arange(n_time) * dt
    trace = np.arange(n_trace)

    fig, ax = plt.subplots(figsize=(6, 8))

    plt.imshow(
        img,
        cmap='gray',
        aspect='auto',
        vmin=-1, vmax=1,                  # clim
        extent=[trace.min(), trace.max(), time.max(), time.min()],  # y轴向下
        interpolation='bicubic'           # bicubic 平滑
    )
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()  # 刻度也放上面
    plt.xlabel('Trace', fontweight='bold', fontsize='16')
    plt.ylabel('Time (s)', fontweight='bold', fontsize='16')

    # y-axis ticks every 0.4s
    yticks = np.arange(0, 1.88, 0.4)
    ax.set_yticks(yticks)

    # x-axis ticks every 50 traces
    xticks = np.arange(0, 256, 50)
    ax.set_xticks(xticks)
    # 刻度数字加粗
    ax.tick_params(axis='x', labelsize=10, labelrotation=0, width=1, length=5, labelcolor='black', direction='out')
    ax.tick_params(axis='y', labelsize=10, width=1, length=5, labelcolor='black', direction='out')
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    # plt.title(name)

    # plt.colorbar()

    save_path = save_dir + name + '.png'
    print(f'saved to: {save_path}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dname", type=str, default='',
                    help='(necessary) name of the data to be tested without the extension, with \'.mat\' as the default format')
parser.add_argument("--dt", type=float, default='',
                    help='(necessary) sampling time interval of this data')

opt = parser.parse_args()

# path setting
fpath = './results/2d/'
methods = [
    'DMINet128',
    'Transformer128',
    'UNet',
          ]
# saving directory
save_dir = f'./figures/{opt.dname}/'
os.makedirs(save_dir, exist_ok=True)

# read data
for method in methods:
    if not os.path.exists(fpath + method + '/'):
        continue

    print(f'method: {method}')
    dpath = fpath + method + f'/for_testing/{opt.dname}.mat'
    mat_data = sio.loadmat(dpath)

    # get data
    datasets = {
        'complete': mat_data['complete'],
        'missing': mat_data['missing'],
        'reconstructed': mat_data['reconstructed'],
        'difference': mat_data['difference']
    }

    for name, img in datasets.items():
        plot_and_save(img, method + '_' + name, opt.dt)

print(f"All images of {opt.dname} data saved to:", save_dir)
