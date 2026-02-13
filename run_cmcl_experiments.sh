#!/bin/bash

# Example training script with Cross-Modal Consistency Loss (CMCL)
# This script demonstrates how to train SiFormer with CMCL on WLASL100 dataset

# ============================================================================
# Configuration 1: Baseline with standard Cross-Entropy
# ============================================================================
echo "Training baseline model with Cross-Entropy..."
python train.py \
    --experiment_name "wlasl100_baseline" \
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" \
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" \
    --num_classes 100 \
    --batch_size 24 \
    --epochs 100 \
    --lr 0.0001 \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --use_cross_attention True \
    --cross_attn_heads 3 \
    --use_cmcl False

# ============================================================================
# Configuration 2: CMCL with balanced hyperparameters (RECOMMENDED)
# ============================================================================
echo "Training with CMCL (balanced)..."
python train.py \
    --experiment_name "wlasl100_cmcl_balanced" \
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" \
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" \
    --num_classes 100 \
    --batch_size 24 \
    --epochs 100 \
    --lr 0.0001 \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --use_cross_attention True \
    --cross_attn_heads 3 \
    --use_cmcl True \
    --lambda_consistency 0.1 \
    --lambda_alignment 0.05 \
    --cmcl_temperature 1.0

# ============================================================================
# Configuration 3: Adaptive CMCL with dynamic weight scheduling
# ============================================================================
echo "Training with Adaptive CMCL..."
python train.py \
    --experiment_name "wlasl100_cmcl_adaptive" \
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" \
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" \
    --num_classes 100 \
    --batch_size 24 \
    --epochs 100 \
    --lr 0.0001 \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --use_cross_attention True \
    --cross_attn_heads 3 \
    --use_cmcl True \
    --lambda_consistency 0.15 \
    --lambda_alignment 0.08 \
    --cmcl_temperature 1.0 \
    --use_adaptive_cmcl True

# ============================================================================
# Configuration 4: Conservative CMCL (for stable training)
# ============================================================================
echo "Training with Conservative CMCL..."
python train.py \
    --experiment_name "wlasl100_cmcl_conservative" \
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" \
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" \
    --num_classes 100 \
    --batch_size 24 \
    --epochs 100 \
    --lr 0.0001 \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --use_cross_attention True \
    --cross_attn_heads 3 \
    --use_cmcl True \
    --lambda_consistency 0.05 \
    --lambda_alignment 0.03 \
    --cmcl_temperature 1.0

# ============================================================================
# Ablation Study 1: Only Consistency Loss
# ============================================================================
echo "Ablation: Only consistency loss..."
python train.py \
    --experiment_name "wlasl100_cmcl_consistency_only" \
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" \
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" \
    --num_classes 100 \
    --batch_size 24 \
    --epochs 100 \
    --lr 0.0001 \
    --use_cross_attention True \
    --use_cmcl True \
    --lambda_consistency 0.1 \
    --lambda_alignment 0.0

# ============================================================================
# Ablation Study 2: Only Alignment Loss
# ============================================================================
echo "Ablation: Only alignment loss..."
python train.py \
    --experiment_name "wlasl100_cmcl_alignment_only" \
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" \
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" \
    --num_classes 100 \
    --batch_size 24 \
    --epochs 100 \
    --lr 0.0001 \
    --use_cross_attention True \
    --use_cmcl True \
    --lambda_consistency 0.0 \
    --lambda_alignment 0.05

echo "All experiments completed!"
