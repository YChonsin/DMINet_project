# run U-Net training using the following command (you can modify the parameters yourself)
# Here is an implementation with one GPU (with a memory of equal or greater than 22GB),
# for the reproducers may not have multiple GPUs.
# If the reproducers have GPUs more than one, you can modify the option '--gpunum' below,
# and quadruple the number after'--batchSize'.
python ./src/reconstruction_train.py --net UNet --epochs 50 --gpunum 1 --lr 0.001 --dpath recon128 --batchSize 16
