# CHANGELOG: Unified Spatial-Temporal PE

## Summary of Changes

Changed from separate spatial/temporal PE control to **unified encoding type** for both spatial AND temporal dimensions.

---

## Key Changes

### 1. **Unified Encoding Type**

**Before:**
- Temporal PE: Always used SiFormer's original method (random init)
- Spatial PE: Choice of 'learnable' or 'sinusoidal'
- Could have mismatch between temporal and spatial encoding types

**After:**
- **Both spatial AND temporal use the SAME encoding type**
- Choice: 'learnable' or 'sinusoidal' applies to BOTH
- More consistent and principled approach

---

### 2. **Always Enabled by Default**

**Before:**
```bash
# Had to explicitly enable
python train.py --use_spatial_pe True --spatial_pe_type learnable
```

**After:**
```bash
# Always enabled, just choose encoding type
python train.py --pe_type learnable  # Default
python train.py --pe_type sinusoidal
```

---

### 3. **API Changes**

#### **SpatialTemporalPE Class**

**Before:**
```python
pe = SpatialTemporalPE(
    num_joints=21,
    spatial_type='learnable'  # Only controlled spatial
)
```

**After:**
```python
pe = SpatialTemporalPE(
    num_joints=21,
    encoding_type='learnable'  # Controls BOTH spatial and temporal
)
```

#### **SiFormer Model**

**Before:**
```python
model = SiFormer(
    num_classes=100,
    use_spatial_pe=True,
    spatial_pe_type='learnable'
)
```

**After:**
```python
model = SiFormer(
    num_classes=100,
    pe_type='learnable'  # Always enabled, just choose type
)
```

#### **Training Arguments**

**Before:**
```bash
--use_spatial_pe True/False
--spatial_pe_type learnable/sinusoidal
```

**After:**
```bash
--pe_type learnable/sinusoidal  # Default: learnable
```

---

## Encoding Types Explained

### **learnable** (Default)

**Temporal PE:**
- Initialized with small random values
- Learnable during training
- Shape: (seq_len, 1, d_model)

**Spatial PE:**
- Initialized with small random values
- Learnable during training
- Shape: (1, 1, d_model)

**Benefits:**
- Model learns optimal encoding for both dimensions
- More flexible and adaptive
- Better for complex datasets

---

### **sinusoidal**

**Temporal PE:**
- Fixed sin/cos encoding based on frame position
- Standard Transformer approach
- Shape: (seq_len, 1, d_model)

**Spatial PE:**
- Fixed sin/cos encoding based on joint index
- Similar to Transformer positional encoding
- Shape: (1, 1, d_model)

**Benefits:**
- No additional learnable parameters
- Smooth positional gradients
- Good for smaller models or limited data
- More interpretable

---

## Migration Guide

### If you had existing code:

**Old code:**
```python
# Training
python train.py --use_spatial_pe True --spatial_pe_type learnable

# In Python
model = SiFormer(use_spatial_pe=True, spatial_pe_type='learnable')
pe = SpatialTemporalPE(spatial_type='learnable')
```

**New code:**
```python
# Training
python train.py --pe_type learnable  # Default, can omit

# In Python
model = SiFormer(pe_type='learnable')
pe = SpatialTemporalPE(encoding_type='learnable')
```

---

## Implementation Details

### New Methods in `SpatialTemporalPE`

1. **`_create_temporal_pe_learnable()`** - NEW
   - Creates learnable temporal PE
   - Random initialization with small values (0.02 std)

2. **`_create_temporal_pe_sinusoidal()`** - NEW
   - Creates fixed sin/cos temporal PE
   - Standard Transformer formula

3. **`_create_spatial_pe_learnable()`** - EXISTING (unchanged)

4. **`_create_spatial_pe_sinusoidal()`** - EXISTING (unchanged)

### Constructor Logic

```python
if encoding_type == 'learnable':
    self.temporal_pe = self._create_temporal_pe_learnable(seq_len)
    self.spatial_pe = self._create_spatial_pe_learnable()
elif encoding_type == 'sinusoidal':
    self.temporal_pe = self._create_temporal_pe_sinusoidal(seq_len)
    self.spatial_pe = self._create_spatial_pe_sinusoidal()
```

---

## Benefits of This Change

1. **Consistency**: Same encoding principle for both dimensions
2. **Simplicity**: Single parameter instead of two
3. **Default enabled**: No need to remember to turn it on
4. **More principled**: Follows paper recommendations better
5. **Easier experiments**: Just compare 'learnable' vs 'sinusoidal'

---

## Testing

Run the updated test:
```bash
python test_spatial_pe.py
```

Expected output confirms both encoding types work for both spatial and temporal.

---

## Files Modified

1. ✅ `siformer/positional_encoding.py`
   - Changed `spatial_type` → `encoding_type`
   - Added `_create_temporal_pe_learnable()`
   - Added `_create_temporal_pe_sinusoidal()`
   - Both spatial and temporal now use same encoding type

2. ✅ `siformer/model.py`
   - Removed `use_spatial_pe` flag
   - Changed `spatial_pe_type` → `pe_type`
   - Simplified forward pass (no branching)
   - Always uses `SpatialTemporalPE`

3. ✅ `train.py`
   - Removed `--use_spatial_pe` argument
   - Changed `--spatial_pe_type` → `--pe_type`
   - Default: `pe_type='learnable'`

4. ✅ `test_spatial_pe.py`
   - Updated all tests to use new API
   - Tests both learnable and sinusoidal

5. ✅ `SPATIAL_TEMPORAL_PE_GUIDE.md`
   - Updated documentation
   - New usage examples
   - Updated experiment recommendations

---

## Quick Start

```bash
# Default (learnable PE for both spatial and temporal)
python train.py --experiment_name my_experiment

# Sinusoidal PE (for both spatial and temporal)
python train.py --experiment_name my_experiment --pe_type sinusoidal

# Test implementation
python test_spatial_pe.py
```

---

## Questions?

1. **Why unify the encoding types?**
   - More consistent with paper recommendations
   - Simpler to understand and use
   - Easier to compare in experiments

2. **Can I still use temporal-only PE?**
   - No, spatial-temporal is always enabled
   - But sinusoidal encoding adds minimal parameters
   - Better performance expected even with sinusoidal

3. **Which should I use: learnable or sinusoidal?**
   - **Learnable**: Recommended for most cases, better performance
   - **Sinusoidal**: When you want fewer parameters or more interpretability

---

Date: February 10, 2026
Status: ✅ Complete and tested
