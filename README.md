# Landmark-Pvt

## HARDWARE: (The following specs were used to create the original solution)
Ubuntu 18.04.3 LTS
Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (80 cores)
8 x NVIDIA Tesla V100 32G

## SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.6.10
CUDA Version 11.0.194
nvidia Driver Version: 418.116.00

## Data preparation

1. Download the Google Landmarks Dataset v2 to `./data` using the scripts at https://github.com/cvdfoundation/google-landmark This is our training data.

2. Download the label csv file at https://s3.amazonaws.com/google-landmark/metadata/train.csv and put it in the same directory as `train` folder

3. Download ReXNet_V1-2.0x pretrained model weights from https://github.com/clovaai/rexnet and put it in `./rexnetv1_2.0x.pth`

4. Run `python preprocess.py` It will read `./train.csv`, create folds and save `./train_0.csv` for training, and save `./idx2landmark_id.pkl` to be used by the submission kernel.


## Training

Training commands of the 9 models.

After training, models will be saved in `./weights/` Tranning logs will be saved in `./logs/` by default.

```
data_dir=./data
model_dir=./weights

### B7 

python -u -m torch.distributed.launch --nproc_per_node=6 train.py --kernel-type b7ns_DDP_final_256_300w_f0_10ep --train-step 0 --data-dir ${data_dir} --image-size 256 --batch-size 42 --enet-type tf_efficientnet_b7_ns --n-epochs 10 --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5 --fold 0 

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b7ns_DDP_final_512_300w_f0_40ep --train-step 1 --data-dir ${data_dir} --image-size 512 --batch-size 16 --enet-type tf_efficientnet_b7_ns --n-epochs 40 --stop-at-epoch 13 --fold 0 --load-from ${model_dir}/b7ns_DDP_final_256_300w_f0_10ep_fold0.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b7ns_final_672_300w_f0_load13_ep20 --train-step 2 --data-dir ${data_dir} --init-lr 5e-5 --image-size 672 --batch-size 10 --enet-type tf_efficientnet_b7_ns --n-epochs 20 --stop-at-epoch 1 --fold 0 --load-from ${model_dir}/b7ns_DDP_final_512_300w_f0_40ep_fold0.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b7ns_final_672_300w_f0_load13_load1_14ep --train-step 3 --data-dir ${data_dir} --init-lr 0.00001 --image-size 672 --batch-size 10 --enet-type tf_efficientnet_b7_ns --n-epochs 14 --stop-at-epoch 4 --fold 0 --load-from ${model_dir}/b7ns_final_672_300w_f0_load13_ep20_fold0.pth

### B6

python -u -m torch.distributed.launch --nproc_per_node=4 train.py --kernel-type b6ns_DDP_final_256_300w_f1_10ep --train-step 0 --data-dir ${data_dir} --image-size 256 --batch-size 64 --enet-type tf_efficientnet_b6_ns --n-epochs 10 --fold 1 --CUDA_VISIBLE_DEVICES 0,1,2,3

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b6ns_DDP_final_512_300w_f1_40ep --train-step 1 --data-dir ${data_dir} --image-size 512 --batch-size 22 --enet-type tf_efficientnet_b6_ns --n-epochs 40 --stop-at-epoch 28 --fold 1 --load-from ${model_dir}/b6ns_DDP_final_256_300w_f1_10ep_fold1.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b6ns_final_768_300w_f1_load28_5ep_1e-5 --train-step 2 --data-dir ${data_dir} --init-lr 1e-5 --image-size 768 --batch-size 10 --enet-type tf_efficientnet_b6_ns --n-epochs 5 --fold 1 --load-from ${model_dir}/b6ns_DDP_final_512_300w_f1_40ep_fold1.pth

### B6

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b6ns_DDP_final_512_300w_f1_40ep --train-step 1 --data-dir ${data_dir} --image-size 512 --batch-size 22 --enet-type tf_efficientnet_b6_ns --n-epochs 40 --stop-at-epoch 36 --fold 1 --load-from ${model_dir}/b6ns_DDP_final_256_300w_f1_10ep_fold1.pth

### B5

python -u -m torch.distributed.launch --nproc_per_node=4 train.py --kernel-type b5ns_DDP_final_256_300w_f2_10ep --train-step 0 --data-dir ${data_dir} --image-size 256 --batch-size 64 --enet-type tf_efficientnet_b5_ns --n-epochs 10 --fold 2 --CUDA_VISIBLE_DEVICES 0,1,2,3

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b5ns_DDP_final_576_300w_f2_40ep --train-step 1 --data-dir ${data_dir} --image-size 576 --batch-size 24 --enet-type tf_efficientnet_b5_ns --n-epochs 40 --stop-at-epoch 16 --fold 2 --load-from ${model_dir}/b5ns_DDP_final_256_300w_f2_10ep_fold2.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b5ns_final_768_300w_f2_load16_20ep --train-step 2 --data-dir ${data_dir} --init-lr 5e-5 --image-size 768 --batch-size 13 --enet-type tf_efficientnet_b5_ns --n-epochs 20 --stop-at-epoch 1 --fold 2 --load-from ${model_dir}/b5ns_DDP_final_576_300w_f2_40ep_fold2.pth

### B5

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b5ns_DDP_final_576_300w_f2_40ep --train-step 1 --data-dir ${data_dir} --image-size 576 --batch-size 24 --enet-type tf_efficientnet_b5_ns --n-epochs 40 --stop-at-epoch 33 --fold 2 --load-from ${model_dir}/b5ns_DDP_final_256_300w_f2_10ep_fold2.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b5ns_final_768_300w_f2_load33_5ep_3e-5_32G --train-step 2 --data-dir ${data_dir} --init-lr 3e-5 --image-size 768 --batch-size 13 --enet-type tf_efficientnet_b5_ns --n-epochs 5 --fold 2 --load-from ${model_dir}/b5ns_DDP_final_576_300w_f2_40ep_fold2.pth

### B4

python -u -m torch.distributed.launch --nproc_per_node=2 train.py --kernel-type b4ns_final_256_400w_f0_10ep --train-step 0 --data-dir ${data_dir} --image-size 256 --batch-size 128 --enet-type tf_efficientnet_b4_ns --n-epochs 10 --fold 0 --CUDA_VISIBLE_DEVICES 0,1

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b4ns_DDP_final_704_300w_f0_50ep --train-step 1 --data-dir ${data_dir} --image-size 704 --batch-size 22 --enet-type tf_efficientnet_b4_ns --n-epochs 50 --stop-at-epoch 16 --fold 0 --load-from ${model_dir}/b4ns_final_256_400w_f0_10ep_fold0.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b4ns_final_768_300w_f0_load16_20ep --train-step 2 --data-dir ${data_dir} --init-lr 5e-5 --image-size 768 --batch-size 16 --enet-type tf_efficientnet_b4_ns --n-epochs 20 --stop-at-epoch 1 --fold 0 --load-from ${model_dir}/b4ns_DDP_final_704_300w_f0_50ep_fold0_ep16.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b4ns_final_768_300w_f0_load16_20ep --train-step 3 --data-dir ${data_dir} --image-size 768 --batch-size 16 --enet-type tf_efficientnet_b4_ns --n-epochs 20 --stop-at-epoch 4 --fold 0 --load-from ${model_dir}/b4ns_final_768_300w_f0_load16_20ep_fold0_ep1.pth

### B3

python -u -m torch.distributed.launch --nproc_per_node=2 train.py --kernel-type b3ns_final_256_400w_f1_10ep --train-step 0 --data-dir ${data_dir} --image-size 256 --batch-size 128 --enet-type tf_efficientnet_b3_ns --n-epochs 10 --fold 1 --CUDA_VISIBLE_DEVICES 0,1

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b3ns_DDP_final_544_300w_f1_40ep --train-step 1 --data-dir ${data_dir} --image-size 544 --batch-size 17 --enet-type tf_efficientnet_b3_ns --n-epochs 40 --stop-at-epoch 29 --fold 1 --load-from ${model_dir}/b3ns_final_256_400w_f1_10ep_fold1.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b3ns_final_768_300w_f1_load29_5ep --train-step 2 --data-dir ${data_dir} --init-lr 5e-5 --image-size 768 --batch-size 22 --enet-type tf_efficientnet_b3_ns --n-epochs 5 --fold 1 --load-from ${model_dir}/b3ns_DDP_final_544_300w_f1_40ep_fold1_ep29.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type b3ns_final_768_300w_f1_load29_5ep5ep --train-step 3 --data-dir ${data_dir} --init-lr 5e-5 --image-size 768 --batch-size 22 --enet-type tf_efficientnet_b3_ns --n-epochs 5 --fold 1 --load-from ${model_dir}/b3ns_final_768_300w_f1_load29_5ep_fold1_full_ep5.pth

### ResNeSt-101

python -u -m torch.distributed.launch --nproc_per_node=6 train.py --kernel-type nest101_DDP_final_256_300w_f4_10ep_3e-5 --train-step 0 --data-dir ${data_dir} --init-lr 3e-5 --image-size 256 --batch-size 42 --enet-type nest101 --n-epochs 10 --fold 4 --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type nest101_DDP_final_576_300w_f4_40ep --train-step 1 --data-dir ${data_dir} --init-lr 3e-5 --image-size 576 --batch-size 30 --enet-type nest101 --n-epochs 40 --stop-at-epoch 16 --fold 4 --load-from ${model_dir}/nest101_DDP_final_256_300w_f4_10ep_3e-5_fold4.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type nest101_final_768_300w_f4_load16_ep20 --train-step 2 --data-dir ${data_dir} --init-lr 2e-5 --image-size 768 --batch-size 16 --enet-type nest101 --n-epochs 20 --stop-at-epoch 1 --fold 4 --load-from ${model_dir}/nest101_DDP_final_576_300w_f4_40ep_fold4.pth

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type nest101_final_768_300w_f4_load16_19ep_load1_16ep --train-step 3 --data-dir ${data_dir} --init-lr 4e-5 --image-size 768 --batch-size 16 --enet-type nest101 --n-epochs 16 --stop-at-epoch 5 --fold 4 --load-from ${model_dir}/nest101_final_768_300w_f4_load16_ep20_fold4.pth

### ReXNet 2.0

python -u -m torch.distributed.launch --nproc_per_node=4 train.py --kernel-type rex20_final_256_400w_f4_10ep --train-step 0 --data-dir ${data_dir} --image-size 256 --batch-size 64 --enet-type rex20 --n-epochs 10 --fold 4 --CUDA_VISIBLE_DEVICES 0,1,2,3

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --kernel-type rex20_DDP_final_768_300w_f4_50ep --train-step 1 --data-dir ${data_dir} --image-size 768 --batch-size 19 --enet-type rex20 --n-epochs 50 --stop-at-epoch 38 --fold 4 --load-from ${model_dir}/rex20_DDP_final_768_300w_f4_50ep_fold4.pth


```

## Predicting

This competition was a code competition. Teams submitted inference notebooks which were ran on hidden test sets. We made public the submission notebook on Kaggle at https://www.kaggle.com/boliu0/landmark-recognition-2020-third-place-submission

All the trained models are linked in that notebook as public datasets. The same notebook is also included in this repo for reference.

## paper

https://arxiv.org/abs/2010.05350