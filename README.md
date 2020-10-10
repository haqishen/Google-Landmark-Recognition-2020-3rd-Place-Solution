# Landmark-Pvt

# Training

Training commands of the 9 models.

After training, models will be saved in `./weights/` Tranning logs will be saved in `./logs/` by default.

```
python train.py --kernel-type b6ns_DDP_final_256_300w_f1_10ep --train-step 0 --data-dir ./data/ --image-size 256 --batch-size 64 --enet-type tf_efficientnet_b6_ns --n-epochs 10 --CUDA_VISIBLE_DEVICES 0 --fold 1

python train.py --kernel-type b6ns_DDP_final_512_300w_f1_40ep --train-step 1 --data-dir ./data/ --image-size 512 --batch-size 22 --enet-type tf_efficientnet_b6_ns --n-epochs 40 --stop-at-epoch 20 --CUDA_VISIBLE_DEVICES 0 --fold 1 --load-from weights/b6ns_DDP_final_256_300w_f1_10ep_fold1.pth

python train.py --kernel-type b6ns_final_768_300w_f1_load28_5ep_1e-5 --train-step 2 --data-dir ./data/ --init-lr 0.00001 --image-size 768 --batch-size 10 --enet-type tf_efficientnet_b6_ns --n-epochs 10 --stop-at-epoch 20 --CUDA_VISIBLE_DEVICES 0 --fold 1 --load-from weights/b6ns_DDP_final_256_300w_f1_10ep_fold1.pth
```