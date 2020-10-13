# Landmark-Pvt

## Pre-processing

1. Download the Google Landmarks Dataset v2 using the scripts at https://github.com/cvdfoundation/google-landmark This is our training data.

2. Download the label csv file at https://s3.amazonaws.com/google-landmark/metadata/train.csv and put it in the same directory as `train` folder

2. Run 
```
python preprocess.py
```
It will read `./train.csv`, create folds and save `./train_0.csv` for training, and save `./idx2landmark_id.pkl` to be used by the submission kernel.


## Training

Training commands of the 9 models.

After training, models will be saved in `./weights/` Tranning logs will be saved in `./logs/` by default.

```

python train.py --kernel-type b7ns_DDP_final_256_300w_f0_10ep --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 32 --enet-type tf_efficientnet_b7_ns --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 0

python train.py --kernel-type b7ns_DDP_final_512_300w_f0_40ep --train-step 1 --data-dir ./data/ --image-size 512 --batch-size 16 --enet-type tf_efficientnet_b7_ns --n-epochs 30 --stop-at-epoch 13 --CUDA_VISIBLE_DEVICES 0 --fold 0 --load-from weights/b7ns_DDP_final_256_300w_f0_10ep_fold0.pth

python train.py --kernel-type b7ns_final_672_300w_f0_load13_load1_14ep --train-step 2 --data-dir ./data/ --init-lr 0.00001 --image-size 672 --batch-size 10 --enet-type tf_efficientnet_b7_ns --n-epochs 14 --CUDA_VISIBLE_DEVICES 0 --fold 0 --load-from weights/b7ns_DDP_final_512_300w_f0_40ep_fold0.pth

###

python train.py --kernel-type b6ns_DDP_final_256_300w_f1_10ep --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 32 --enet-type tf_efficientnet_b6_ns --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 1

python train.py --kernel-type b6ns_DDP_final_512_300w_f1_40ep --train-step 1 --data-dir ./data/ --image-size 512 --batch-size 22 --enet-type tf_efficientnet_b6_ns --n-epochs 40 --stop-at-epoch 28 --CUDA_VISIBLE_DEVICES 0 --fold 1 --load-from weights/b6ns_DDP_final_256_300w_f1_10ep_fold1.pth

python train.py --kernel-type b6ns_final_768_300w_f1_load28_5ep_1e-5 --train-step 2 --data-dir ./data/ --init-lr 0.00001 --image-size 768 --batch-size 10 --enet-type tf_efficientnet_b6_ns --n-epochs 10 --stop-at-epoch 5 --CUDA_VISIBLE_DEVICES 0 --fold 1 --load-from weights/b6ns_DDP_final_512_300w_f1_40ep_fold1.pth

###

python train.py --kernel-type b5ns_DDP_final_256_300w_f2_10ep --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 32 --enet-type tf_efficientnet_b5_ns --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 2

python train.py --kernel-type b5ns_DDP_final_576_300w_f2_40ep --train-step 1 --data-dir ./data/ --image-size 576 --batch-size 24 --enet-type tf_efficientnet_b5_ns --n-epochs 40 --stop-at-epoch 33 --CUDA_VISIBLE_DEVICES 0 --fold 2 --load-from weights/b5ns_DDP_final_256_300w_f2_10ep_fold2.pth

python train.py --kernel-type b5ns_final_768_300w_f2_load33_5ep_3e-5_32G --train-step 2 --data-dir ./data/ --init-lr 0.00003 --image-size 768 --batch-size 13 --enet-type tf_efficientnet_b5_ns --n-epochs 5 --CUDA_VISIBLE_DEVICES 0 --fold 2 --load-from weights/b5ns_DDP_final_576_300w_f2_40ep_fold2.pth


###

python train.py --kernel-type b4ns_final_256_400w_f0_10ep --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 32 --enet-type tf_efficientnet_b4_ns --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 0

python train.py --kernel-type b4ns_DDP_final_704_300w_f0_50ep --train-step 1 --data-dir ./data/ --image-size 704 --batch-size 22 --enet-type tf_efficientnet_b4_ns --n-epochs 50 --stop-at-epoch 16 --CUDA_VISIBLE_DEVICES 0 --fold 0 --load-from weights/b4ns_final_256_400w_f0_10ep_fold0.pth

python train.py --kernel-type b4ns_final_768_300w_f0_load16_20ep_load1_20ep --train-step 2 --data-dir ./data/ --init-lr 0.00001 --image-size 768 --batch-size 13 --enet-type tf_efficientnet_b4_ns --n-epochs 5 --CUDA_VISIBLE_DEVICES 0 --fold 0 --load-from weights/b4ns_DDP_final_704_300w_f0_50ep_fold0.pth

###

python train.py --kernel-type b3ns_final_256_400w_f1_10ep --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 32 --enet-type tf_efficientnet_b3_ns --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 1

python train.py --kernel-type b3ns_DDP_final_544_300w_f1_40ep --train-step 1 --data-dir ./data/ --image-size 544 --batch-size 22 --enet-type tf_efficientnet_b3_ns --n-epochs 50 --stop-at-epoch 16 --CUDA_VISIBLE_DEVICES 0 --fold 1 --load-from weights/b3ns_final_256_400w_f1_10ep_fold1.pth

python train.py --kernel-type b3ns_final_768_300w_f1_load29_5ep5ep --train-step 2 --data-dir ./data/ --init-lr 0.00005 --image-size 768 --batch-size 13 --enet-type tf_efficientnet_b3_ns --n-epochs 5 --CUDA_VISIBLE_DEVICES 0 --fold 1 --load-from weights/b3ns_DDP_final_544_300w_f1_40ep_fold1.pth

###

python train.py --kernel-type nest101_DDP_final_256_300w_f4_10ep_3e-5 --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 32 --enet-type nest101 --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 4

python train.py --kernel-type nest101_DDP_final_576_300w_f4_40ep --train-step 1 --data-dir ./data/ --image-size 544 --batch-size 22 --enet-type nest101 --n-epochs 40 --stop-at-epoch 16 --CUDA_VISIBLE_DEVICES 0 --fold 4 --load-from weights/b3ns_final_256_400w_f1_10ep_fold4.pth

python train.py --kernel-type nest101_final_768_300w_f4_load16_19ep_load1_16ep --train-step 2 --data-dir ./data/ --init-lr 0.00005 --image-size 768 --batch-size 13 --enet-type nest101 --n-epochs 5 --CUDA_VISIBLE_DEVICES 0 --fold 4 --load-from weights/b3ns_DDP_final_544_300w_f1_40ep_fold4.pth


###

python train.py --kernel-type rex20_final_256_400w_f4_10ep --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 32 --enet-type nest101 --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 4

python train.py --kernel-type rex20_DDP_final_768_300w_f4_50ep --train-step 1 --data-dir ./data/ --image-size 544 --batch-size 22 --enet-type nest101 --n-epochs 50 --stop-at-epoch 21 --CUDA_VISIBLE_DEVICES 0 --fold 4 --load-from weights/rex20_final_256_400w_f4_10ep_fold4.pth

python train.py --kernel-type rex20_DDP_final_768_300w_f4_35ep_load20resume --train-step 2 --data-dir ./data/ --init-lr 0.00005 --image-size 768 --batch-size 13 --enet-type nest101 --n-epochs 5 --CUDA_VISIBLE_DEVICES 0 --fold 4 --load-from weights/rex20_DDP_final_768_300w_f4_50ep_fold4.pth

```

## Predicting

This competition was a code competition. Teams submitted inference notebooks which were ran on hidden test sets. We made public the submission notebook on Kaggle at https://www.kaggle.com/boliu0/landmark-recognition-2020-third-place-submission

All the trained models are linked in that notebook as public datasets. The same notebook is also included in this repo for reference.