
# A Globally Oriented and Lightweight Seismic Interpolation Network: More Global than Convolutional Neural Network and More Efficient than Transformer

## Introduction
Code implementation of DMI-Net and its training and testing process.

This repository was completed on Mar. 24, 2026 by Changxin Wei, 
a PhD candidate at the College of Instrumentation and Electrical Engineering, Jilin University，
Changchun, China.

If there are any problems with this project, please feel free to contact cxwei24@mails.jlu.edu.cn, 
or you can ask ChatGPT for help.

**Attention**

**The '.mat' files provided here are only examples that help to quickly train and test DMI-Net.**

All the provided seismic datasets are saved in '.mat' format for simplicity. If you intend to use your 
own datasets, please make sure the filename is consistent with the variable name inside the '.mat' file.

As for test data, **only** synthetic data, which corresponds to that in the paper, is provided as an example. 
The field data in the paper can not be shared, and you can search for other public available field datasets for 
testing. All test data should be saved in **'./data/for_testing'**.

---

## File Introduction
```
DMINet_project
│
├── README.md                       # read this first
├── requirements.txt                # used for package installation
├── __init__.py                                 
│
├── src/
│   ├── reconstruction_train.py     # training
│   ├── reconstruction_test.py      # testing
│   ├── dataset_processing.py       # dataset processing
│   ├── utils.py                    # useful functions
│   ├── generate_figures.py         # to generate figures in the paper
│   ├── __init__.py         
│   └── model/
│       ├── TIR.py                  # main code of Transformer (Competitive method)
│       ├── UNet.py                 # main code of U-Net (Competitive method)
│       ├── DMINet.py               # main code of DMI-Net
│       ├── Transformer128.py       # an object of Transformer
│       ├── DMINet128.py            # an object of DMI-Net
│       └── __init__.py
│
├── scripts/                        # shell scripts
│   ├── dataset_processing.sh       # process the original data into patches
│   ├── run_train_DMINet.sh         # run training of DMI-Net
│   ├── run_train_Transformer.sh    # run training of Transformer
│   ├── run_train_UNet.sh           # run training of U-Net
│   ├── run_test_DMINet.sh          # run testing of DMI-Net
│   ├── run_test_Transformer.sh     # run testing of Transformer
│   ├── run_test_UNet.sh            # run testing of U-Net
│   └── generate_figures.sh         # plot the results and save them
│
├── data/                           # original data, not processed (saved in '.mat' format)
│   ├── for_training                # data for training
│   │   └── ...                     
│   └── for_testing                 # data for testing
│       └── ...                     
│
├── prepared_dataset/               # dataset used for training
│   ├── train/                      # training set
│   │   ├── label/                  # complete seismic data patches
│   │   │   ├── 1.npy               # 1.npy, 2.npy, ..., N.npy,  
│   │   │   └── ....npy             # where N is the size of dataset    
│   │   └── feature/                # corresponding incomplete seismic data patches
│   │       ├── 1.npy               # correspond with those in the label folder above
│   │       └── ....npy                
│   └── val/                        # validation set, the same folder arrangement with train folder
│       ├── label/
│       │   ├── 1.npy               
│       │   └── ....npy               
│       └── feature/
│           ├── 1.npy               
│           └── ....npy     
│
├── trained_models/                 # trained models will be saved here
│   └── 2d/      
│       ├── DMINet_example/         # example models of the three networks
│       │   └── recon_best.pth    
│       ├── Transformer_example/         
│       │   └── recon_best.pth    
│       ├── UNet_example/ 
│       │   └── recon_best.pth    
│       └── ...                   
│  
├── logs/           
│   └── ....log                     # used to record the training process
│  
└── results/                        # interpolated results will be saved here
    └── 2d/
        ├── synthetic/  
        └── field/       
```
---

## How to run training and testing
You need to follow these instructions step-by-step to train and test DMI-Net successfully.
### 1. Environment
A step-by-step tutorial for environment establishment.
#### 1.1 Python installment
If Python already exists in your **Linux-OS** computer (not recommended on Windows), skip to **Step 1.2**; 
otherwise, you should install Python in your Linux machine.\
\
① Here is an example for installing Python using Anaconda.
```shell
# 1. Download Anaconda in any directory (Specific Version
# can be found in the official website: https://www.anaconda.com/)
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

# 2. Setup Anaconda by
bash ./Anaconda3-2024.06-1-Linux-x86_64.sh

# 3. Add Anaconda3's bin path to environment variable
# (You should replace the path in the following command)
echo 'export PATH="/path/to/your/anaconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 4. Test your installation of Anaconda.
conda --version

# 5. Python3 and Pip3 are integrated in Anaconda, test them.
python --version
pip --version
```
You can proceed in the base environment, or create a new environment as below (recommended).

② For the first time you create an environment, you should use the following command:
```shell
conda init
```
③ Reboot your terminal.

④ Create an environment named 'testenv':
```shell
conda create -n testenv python=3.12
```
⑤ Activate 'testenv':
```shell
conda activate testenv
```

#### 1.2 Install all required packages
```shell
pip install -r requirements.txt
```

#### 1.3 Install Mamba
① Check the torch version by:
```shell
pip list
```
This output indicates the version of torch is 2.10.x.
```shell
...
torch                    2.10.0
...
```
② Check the CUDA version by:
```shell
nvcc --version
```

This output indicates the version of CUDA is 12.x.
```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

③ Visit https://github.com/state-spaces/mamba/releases, \
and find a suitable version of mamba_ssm, and copy its downloading link.
```shell
https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
**Here, 'cu12' means CUDA 12.x, 'torch2.10' means torch 2.10.x.**

**Besides, the final part of the release name should be ‘...-linux_x86_64.whl', 
since you're using an x86-64 Linux OS.**

④ Paste the link after command **[pip install]**, and run this command like this:
```shell
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
If the installation process went successfully, you may continue to **Step 2**;\
Otherwise, you should re-check **the versions of torch and CUDA**, and re-select the release
of mamba_ssm very carefully. 

PS: If your versions of torch and CUDA are not shown in the release list of **mamba_ssm-2.3.1**, maybe a 
suitable release can be found in an older version of mamba_ssm. 

-------

### 2. Data preparation
To obtain training and validation datasets, you should run the following command
to process the original seismic data into data patches and save them into dataset folder
in '.npy' format, and all the data patches will be automatically arranged as 
in section **File Introduction** above.

Run the bash script as follows:
``` shell
bash ./scripts/dataset_processing.sh
```
and this indicates a successful processing:
```shell
Training set: 11624 patches
Validation set: 1408 patches
```

-------
### 3. Training
After finishing **Step 1** and **Step 2**, you may start training.

Training for DMI-Net:
```shell
bash ./scripts/run_train_DMINet.sh
```
Training for Transformer:
```shell
bash ./scripts/run_train_Transformer.sh
```
Training for U-Net:
```shell
bash ./scripts/run_train_UNet.sh
```

The trained models will be saved to **'./trained_models/2d/'**.

PS: The parameters in these shell scripts are set for one-GPU training. 
You can modify them for multi-GPU training.

-------
### 4. Testing
After finishing training, test them using:
```shell
bash ./scripts/run_test_DMINet.sh [folder name]
bash ./scripts/run_test_Transformer.sh [folder name]
bash ./scripts/run_test_UNet.sh [folder name]
```
where **[folder name]** should be modified into the base folder name of the trained model.

Here, we provide example models (trained for 20 epochs) of the three networks, saved in './trained_models'.

For example, after running
```shell
bash ./scripts/run_test_UNet.sh UNet_example
```
the interpolated results will be saved to './results/2d/UNet/' in '.mat' format.
There will be four dataset in one '.mat'-format result:

① **complete**:         complete data\
② **missing**:          incomplete data\
③ **reconstructed**:    interpolated result\
④ **difference**:       obtained by subtracting **missing** from **complete**

-------

### 5. Figure Generation
Run this command
```shell
bash ./scripts/generate_figures.sh [data name] [sampling interval]
```
and all the results will be plotted and saved to **'./figures'**.

For example,
```shell
bash ./scripts/generate_figures.sh synthetic 0.002
```

---

