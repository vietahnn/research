#!/bin/bash
# Training script for SiFormer with Cross-Modal Attention experiments
# Author: Research Team
# Date: February 2026

# ============================================
# Configuration
# ============================================
DATASET_DIR="path/to/your/dataset"
TRAIN_CSV="${DATASET_DIR}/train.csv"
TEST_CSV="${DATASET_DIR}/test.csv"
VAL_CSV="${DATASET_DIR}/val.csv"

NUM_CLASSES=100
BATCH_SIZE=24
EPOCHS=100
NUM_WORKERS=24
LR=0.0001

# ============================================
# Experiment 1: Baseline without Cross-Attention
# ============================================
echo "=========================================="
echo "Running Baseline (No Cross-Attention)"
echo "=========================================="

python train.py \
    --experiment_name baseline_no_cross_attn \
    --training_set_path ${TRAIN_CSV} \
    --testing_set_path ${TEST_CSV} \
    --validation_set from-file \
    --validation_set_path ${VAL_CSV} \
    --num_classes ${NUM_CLASSES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_worker ${NUM_WORKERS} \
    --lr ${LR} \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --FIM True \
    --IA_encoder True \
    --IA_decoder False \
    --patience 1 \
    --use_cross_attention False \
    --save_checkpoints True \
    --plot_stats True

# ============================================
# Experiment 2: With Cross-Modal Attention (2 heads)
# ============================================
echo "=========================================="
echo "Running Cross-Attention with 2 heads"
echo "=========================================="

python train.py \
    --experiment_name cross_attn_heads_2 \
    --training_set_path ${TRAIN_CSV} \
    --testing_set_path ${TEST_CSV} \
    --validation_set from-file \
    --validation_set_path ${VAL_CSV} \
    --num_classes ${NUM_CLASSES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_worker ${NUM_WORKERS} \
    --lr ${LR} \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --FIM True \
    --IA_encoder True \
    --IA_decoder False \
    --patience 1 \
    --use_cross_attention True \
    --cross_attn_heads 2 \
    --save_checkpoints True \
    --plot_stats True

# ============================================
# Experiment 3: With Cross-Modal Attention (4 heads - Recommended)
# ============================================
echo "=========================================="
echo "Running Cross-Attention with 4 heads (Recommended)"
echo "=========================================="

python train.py \
    --experiment_name cross_attn_heads_4 \
    --training_set_path ${TRAIN_CSV} \
    --testing_set_path ${TEST_CSV} \
    --validation_set from-file \
    --validation_set_path ${VAL_CSV} \
    --num_classes ${NUM_CLASSES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_worker ${NUM_WORKERS} \
    --lr ${LR} \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --FIM True \
    --IA_encoder True \
    --IA_decoder False \
    --patience 1 \
    --use_cross_attention True \
    --cross_attn_heads 4 \
    --save_checkpoints True \
    --plot_stats True

# ============================================
# Experiment 4: With Cross-Modal Attention (8 heads)
# ============================================
echo "=========================================="
echo "Running Cross-Attention with 8 heads"
echo "=========================================="

python train.py \
    --experiment_name cross_attn_heads_8 \
    --training_set_path ${TRAIN_CSV} \
    --testing_set_path ${TEST_CSV} \
    --validation_set from-file \
    --validation_set_path ${VAL_CSV} \
    --num_classes ${NUM_CLASSES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_worker ${NUM_WORKERS} \
    --lr ${LR} \
    --num_enc_layers 3 \
    --num_dec_layers 2 \
    --FIM True \
    --IA_encoder True \
    --IA_decoder False \
    --patience 1 \
    --use_cross_attention True \
    --cross_attn_heads 8 \
    --save_checkpoints True \
    --plot_stats True

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
