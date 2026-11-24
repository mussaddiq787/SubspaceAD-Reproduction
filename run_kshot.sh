#!/bin/bash
# k-shot ablation: same settings as the k=1 reproduction, only k changes
for k in 2 4
do
    echo "=== Running k=$k ==="
    python -u main.py \
        --dataset_name mvtec_ad \
        --dataset_path datasets/mvtec-ad \
        --image_res 448 \
        --k_shot $k \
        --layers="-12,-13,-14,-15,-16,-17,-18" \
        --model_ckpt "facebook/dinov2-with-registers-large" \
        --aug_count 30 \
        --pca_ev 0.99 \
        --seed 42 \
        --agg_method "mean" \
        --outdir results/repro_k${k}_mvtec_L448
done
echo "=== k-shot ablation done ==="
