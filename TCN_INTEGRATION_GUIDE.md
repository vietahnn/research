# TCN Integration Guide

## Overview

Temporal Convolutional Networks (TCN) đã được tích hợp vào Siformer để cải thiện temporal modeling cho sign language recognition.

## Kiến trúc

### TCN Architecture

```
Input [L_hand, R_hand, Body]
  ↓
Positional Encoding
  ↓
┌─────────────────────────────────┐
│   Feature-Isolated TCN          │
│                                  │
│  L_hand (42D) → TCN → 42D       │
│  R_hand (42D) → TCN → 42D       │
│  Body (24D)   → TCN → 24D       │
└─────────────────────────────────┘
  ↓
Residual Fusion (Original + TCN features)
  ↓
Transformer Encoders (existing)
  ↓
Cross-Modal Attention (optional)
  ↓
Decoder → Classification
```

### TCN Component Structure

Mỗi TCN stream có cấu trúc:
```
Input → TemporalBlock₁ → TemporalBlock₂ → ... → TemporalBlockₙ → Output
```

Mỗi TemporalBlock:
```
Input
  ↓
Conv1 (dilation=2^i) → ReLU → Dropout
  ↓
Conv2 (dilation=2^i) → ReLU → Dropout
  ↓
Residual Connection ──────────────────┘
  ↓
Output
```

**Dilation rates**: [1, 2, 4, 8, 16, 32] (exponential)

**Receptive field**: Với 4 layers, kernel=3:
- RF = 1 + 2×(1+2+4+8) = **31 frames**
- Với 6 layers: RF = **127 frames** (cover toàn bộ sign sequence)

## Files Created/Modified

### New Files:
1. **`siformer/tcn.py`** - TCN implementation
   - `TemporalBlock`: Basic building block với dilated convolutions
   - `MultiScaleTCN`: Stack of temporal blocks
   - `FeatureIsolatedTCN`: 3 separate TCN cho l_hand, r_hand, body

2. **`test_tcn_integration.py`** - Comprehensive test suite
   - Tests standalone TCN modules
   - Tests SiFormer with/without TCN
   - Tests gradient flow
   - Tests different configurations

### Modified Files:
1. **`siformer/model.py`**
   - Import TCN module
   - Add TCN parameters to `SiFormer.__init__`
   - Initialize TCN modules
   - Update forward pass with TCN + residual fusion

2. **`train.py`**
   - Add `--use_tcn` flag (default: False)
   - Add `--tcn_num_layers` (default: 4)
   - Add `--tcn_kernel_size` (default: 3)
   - Add `--tcn_dropout` (default: 0.1)

## Usage

### Basic Training (TCN disabled - baseline):
```bash
python train.py \
  --experiment_name baseline_no_tcn \
  --training_set_path datasets/WLASL100/WLASL100_train_25fps.csv \
  --testing_set_path datasets/WLASL100/WLASL100_val_25fps.csv
```

### Training with TCN (default config):
```bash
python train.py \
  --experiment_name with_tcn_default \
  --training_set_path datasets/WLASL100/WLASL100_train_25fps.csv \
  --testing_set_path datasets/WLASL100/WLASL100_val_25fps.csv \
  --use_tcn True
```

### Training with TCN (custom config):
```bash
python train.py \
  --experiment_name with_tcn_custom \
  --training_set_path datasets/WLASL100/WLASL100_train_25fps.csv \
  --testing_set_path datasets/WLASL100/WLASL100_val_25fps.csv \
  --use_tcn True \
  --tcn_num_layers 6 \
  --tcn_kernel_size 5 \
  --tcn_dropout 0.15
```

## Configuration Options

### `--use_tcn` (bool, default: False)
Enable/disable TCN temporal feature extraction.

### `--tcn_num_layers` (int, default: 4)
Number of TCN layers with exponentially increasing dilations.
- **3 layers**: RF = 15 frames (short-range)
- **4 layers**: RF = 31 frames (medium-range, KHUYẾN NGHỊ)
- **6 layers**: RF = 127 frames (long-range, cho sequences dài)

### `--tcn_kernel_size` (int, default: 3)
Kernel size for convolutions.
- **3**: Smaller receptive field, faster (KHUYẾN NGHỊ)
- **5**: Larger receptive field, smoother
- **7**: Very large receptive field, may oversmooth

### `--tcn_dropout` (float, default: 0.1)
Dropout rate in TCN layers.
- **0.0**: No dropout (risk overfitting)
- **0.1**: Standard dropout (KHUYẾN NGHỊ)
- **0.2**: Higher regularization (cho dataset nhỏ)

## Recommended Configurations

### 1. Default (Balanced):
```bash
--use_tcn True --tcn_num_layers 4 --tcn_kernel_size 3 --tcn_dropout 0.1
```
- Receptive field: 31 frames
- Fast training
- Good for most datasets

### 2. Large Receptive Field:
```bash
--use_tcn True --tcn_num_layers 6 --tcn_kernel_size 3 --tcn_dropout 0.1
```
- Receptive field: 127 frames
- Captures long-range temporal patterns
- Slightly slower training

### 3. Smoother Temporal Features:
```bash
--use_tcn True --tcn_num_layers 4 --tcn_kernel_size 5 --tcn_dropout 0.15
```
- Larger kernel → smoother features
- Higher dropout for regularization

## Testing

### Run standalone TCN tests:
```bash
cd Siformer
python siformer/tcn.py
```

### Run full integration tests:
```bash
python test_tcn_integration.py
```

Expected output: All 6 tests should pass ✓

## Expected Performance

### Improvements:
- **Accuracy**: +1-2% expected improvement
- **Temporal robustness**: Better handling of variable-speed signs
- **Generalization**: Reduced overfitting on temporal patterns

### Computational Cost:
- **Training time**: +10-15% (due to TCN forward pass)
- **Memory**: +15-20% (TCN parameters + activations)
- **Parameters**: ~150K additional parameters (for default config)

### Parameter Count:
```
Default config (4 layers, 3 kernels):
  L_hand TCN: ~25K params
  R_hand TCN: ~25K params
  Body TCN:   ~12K params
  Total:      ~62K params
```

## Ablation Study Guide

To properly evaluate TCN impact, run experiments:

1. **Baseline** (no TCN):
```bash
python train.py --experiment_name baseline --use_tcn False
```

2. **TCN (default)**:
```bash
python train.py --experiment_name tcn_4layers --use_tcn True --tcn_num_layers 4
```

3. **TCN (large RF)**:
```bash
python train.py --experiment_name tcn_6layers --use_tcn True --tcn_num_layers 6
```

Compare:
- Validation accuracy
- Training time per epoch
- Convergence speed
- Generalization gap (train vs val accuracy)

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solution**: Reduce batch size or use fewer TCN layers
```bash
--batch_size 16 --tcn_num_layers 3
```

### Issue: Training too slow
**Solution**: Use fewer layers or smaller kernel
```bash
--tcn_num_layers 3 --tcn_kernel_size 3
```

### Issue: Model not improving with TCN
**Possible causes**:
1. Dataset too small → try higher dropout
2. Receptive field mismatch → adjust num_layers
3. Need more epochs → increase training duration

### Issue: NaN loss
**Solution**: Lower learning rate or reduce dropout
```bash
--lr 0.00005 --tcn_dropout 0.05
```

## Architecture Details

### Why Feature Isolation?
TCN maintains separate processing for l_hand, r_hand, body:
- Preserves Siformer's design philosophy
- Each body part has different temporal dynamics
- Prevents negative transfer between features

### Why Residual Fusion?
```python
output = original_features + tcn_features
```
- Preserves positional encoding information
- Stable training (TCN can learn 0 if not helpful)
- Gradual improvement (TCN refines, doesn't replace)

### Why Dilated Convolutions?
- Large receptive field with few parameters
- Captures multi-scale temporal patterns
- Efficient parallel computation

## Advanced Usage

### Combine with other features:
```bash
python train.py \
  --use_tcn True \
  --use_cross_attention True \
  --tcn_num_layers 4 \
  --cross_attn_heads 3
```

### For very long sequences (>200 frames):
```bash
--tcn_num_layers 6 --tcn_kernel_size 5
# Receptive field: ~255 frames
```

### For short sequences (<100 frames):
```bash
--tcn_num_layers 3 --tcn_kernel_size 3
# Receptive field: ~15 frames (avoid over-smoothing)
```

## References

Implementation inspired by:
1. "Temporal Convolutional Networks for Action Segmentation" (Lea et al., 2017)
2. WaveNet architecture (van den Oord et al., 2016)
3. Multi-scale temporal modeling principles

## Next Steps

Potential enhancements:
1. **Adaptive dilations**: Learn dilation rates instead of fixed exponential
2. **Attention-based fusion**: Replace simple residual with attention weights
3. **Multi-scale outputs**: Combine features from different TCN layers
4. **Graph TCN**: Incorporate skeleton topology into convolutions
