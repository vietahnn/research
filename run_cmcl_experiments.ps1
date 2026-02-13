# PowerShell script for training with Cross-Modal Consistency Loss (CMCL)
# This script demonstrates how to train SiFormer with CMCL on WLASL100 dataset

# ============================================================================
# Configuration 1: Baseline with standard Cross-Entropy
# ============================================================================
Write-Host "Training baseline model with Cross-Entropy..." -ForegroundColor Green
python train.py `
    --experiment_name "wlasl100_baseline" `
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" `
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" `
    --num_classes 100 `
    --batch_size 24 `
    --epochs 100 `
    --lr 0.0001 `
    --num_enc_layers 3 `
    --num_dec_layers 2 `
    --use_cross_attention True `
    --cross_attn_heads 3 `
    --use_cmcl False

# ============================================================================
# Configuration 2: CMCL with balanced hyperparameters (RECOMMENDED)
# ============================================================================
Write-Host "Training with CMCL (balanced)..." -ForegroundColor Green
python train.py `
    --experiment_name "wlasl100_cmcl_balanced" `
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" `
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" `
    --num_classes 100 `
    --batch_size 24 `
    --epochs 100 `
    --lr 0.0001 `
    --num_enc_layers 3 `
    --num_dec_layers 2 `
    --use_cross_attention True `
    --cross_attn_heads 3 `
    --use_cmcl True `
    --lambda_consistency 0.1 `
    --lambda_alignment 0.05 `
    --cmcl_temperature 1.0

# ============================================================================
# Configuration 3: Adaptive CMCL with dynamic weight scheduling
# ============================================================================
Write-Host "Training with Adaptive CMCL..." -ForegroundColor Green
python train.py `
    --experiment_name "wlasl100_cmcl_adaptive" `
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" `
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" `
    --num_classes 100 `
    --batch_size 24 `
    --epochs 100 `
    --lr 0.0001 `
    --num_enc_layers 3 `
    --num_dec_layers 2 `
    --use_cross_attention True `
    --cross_attn_heads 3 `
    --use_cmcl True `
    --lambda_consistency 0.15 `
    --lambda_alignment 0.08 `
    --cmcl_temperature 1.0 `
    --use_adaptive_cmcl True

# ============================================================================
# Quick Test: CMCL with 10 epochs (for quick validation)
# ============================================================================
Write-Host "Quick test with CMCL (10 epochs)..." -ForegroundColor Yellow
python train.py `
    --experiment_name "wlasl100_cmcl_test" `
    --training_set_path "datasets/WLASL100/WLASL100_train_25fps.csv" `
    --testing_set_path "datasets/WLASL100/WLASL100_val_25fps.csv" `
    --num_classes 100 `
    --batch_size 24 `
    --epochs 10 `
    --lr 0.0001 `
    --use_cross_attention True `
    --use_cmcl True `
    --lambda_consistency 0.1 `
    --lambda_alignment 0.05

Write-Host "All experiments completed!" -ForegroundColor Green
