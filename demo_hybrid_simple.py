"""
DEMO: Simple Hybrid Normalization - Approach 4 (Augmented Features)
Cách đơn giản nhất để kết hợp position và shape information
"""

import torch
import torch.nn as nn
import numpy as np

print("="*80)
print("DEMO: HYBRID NORMALIZATION - AUGMENTED FEATURES APPROACH")
print("="*80)

# ============================================================================
# 1. GIẢI THÍCH CONCEPT
# ============================================================================
print("\n📚 CONCEPT:")
print("-" * 80)
print("""
Idea: Giữ nguyên Siformer architecture, chỉ thêm position info vào input

Original Siformer:
    Input: (B, T, 21, 2)  ← Chỉ có x, y (shape only)
    
Hybrid Siformer:
    Input: (B, T, 21, 4)  ← x, y (shape) + rel_x, rel_y (position)
    
    Channel 0-1: Isolated normalized coordinates (shape info)
    Channel 2-3: Relative position to body center (position info)
    
Model tự học khi nào cần dùng shape, khi nào cần dùng position!
""")

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================
print("\n📦 BƯỚC 1: CHUẨN BỊ DỮ LIỆU")
print("-" * 80)

def compute_position_features(raw_landmarks, eyes_idx=168):
    """
    Tính position features cho mỗi bàn tay
    
    Args:
        raw_landmarks: (T, N, 3) - Raw landmark coordinates
        eyes_idx: Index của midway between eyes
        
    Returns:
        left_hand_pos: (T, 2) - Position của left wrist so với eyes
        right_hand_pos: (T, 2) - Position của right wrist so với eyes
    """
    T = raw_landmarks.shape[0]
    
    # Reference point: Midway between eyes
    eyes_center = raw_landmarks[:, eyes_idx, :2]  # (T, 2)
    
    # Wrist positions (giả sử index 0 và 21)
    left_wrist = raw_landmarks[:, 0, :2]   # Left hand wrist
    right_wrist = raw_landmarks[:, 21, :2]  # Right hand wrist
    
    # Relative positions
    left_hand_pos = left_wrist - eyes_center  # (T, 2)
    right_hand_pos = right_wrist - eyes_center
    
    # Normalize to [-1, 1] based on image size (giả sử 640x480)
    left_hand_pos = left_hand_pos / np.array([320, 240])
    right_hand_pos = right_hand_pos / np.array([320, 240])
    
    # Clip to [-1, 1]
    left_hand_pos = np.clip(left_hand_pos, -1, 1)
    right_hand_pos = np.clip(right_hand_pos, -1, 1)
    
    return left_hand_pos, right_hand_pos


def augment_landmarks_with_position(
    shape_landmarks,  # (T, 21, 2) - Siformer normalized
    position_features  # (T, 2) - Relative position
):
    """
    Thêm position features vào mỗi landmark
    
    Returns:
        augmented: (T, 21, 4) - Shape + Position
    """
    T, N, C = shape_landmarks.shape
    
    # Expand position to all landmarks
    # Position của wrist được clone cho tất cả landmarks của bàn tay đó
    position_expanded = np.expand_dims(position_features, axis=1)  # (T, 1, 2)
    position_expanded = np.tile(position_expanded, (1, N, 1))  # (T, 21, 2)
    
    # Concatenate shape + position
    augmented = np.concatenate([shape_landmarks, position_expanded], axis=-1)
    
    return augmented  # (T, 21, 4)


# Demo với dữ liệu giả
T = 100  # 100 frames
N_landmarks = 543  # Total landmarks

# Giả lập raw data
raw_landmarks = np.random.randn(T, N_landmarks, 3) * 100 + 500

# Giả lập Siformer normalized data (đã qua isolated normalization)
left_hand_shape = np.random.rand(T, 21, 2) - 0.5  # [-0.5, 0.5]
right_hand_shape = np.random.rand(T, 21, 2) - 0.5

print(f"✓ Raw landmarks shape: {raw_landmarks.shape}")
print(f"✓ Left hand (shape only): {left_hand_shape.shape}")

# Tính position features
left_pos, right_pos = compute_position_features(raw_landmarks)
print(f"✓ Left hand position: {left_pos.shape}")
print(f"  Sample position at frame 0: {left_pos[0]}")

# Augment với position
left_hand_augmented = augment_landmarks_with_position(left_hand_shape, left_pos)
right_hand_augmented = augment_landmarks_with_position(right_hand_shape, right_pos)

print(f"✓ Left hand (augmented): {left_hand_augmented.shape}")
print(f"  - Channel 0-1: Shape coordinates")
print(f"  - Channel 2-3: Position relative to body")

# ============================================================================
# 3. MODEL MODIFICATION
# ============================================================================
print("\n\n🔧 BƯỚC 2: MODIFY MODEL")
print("-" * 80)

print("""
Chỉ cần thay đổi INPUT DIMENSION từ 2 → 4:

Original SiFormer encoder:
    nn.Linear(2, hidden_dim)  ← 2 channels (x, y)

Hybrid SiFormer encoder:
    nn.Linear(4, hidden_dim)  ← 4 channels (x, y, rel_x, rel_y)

Code changes:
""")

print("""
# In siformer/encoder.py

class SiFormerEncoder(nn.Module):
    def __init__(self, num_hid=108, input_dim=4):  # ← Add input_dim parameter
        super().__init__()
        
        # Embedding layer
        self.embed = nn.Linear(input_dim, num_hid)  # ← Use input_dim instead of 2
        
        # Rest remains the same
        self.transformer_layers = ...
        
    def forward(self, x):
        # x: (B, T, 21, 4) instead of (B, T, 21, 2)
        B, T, N, C = x.shape
        
        # Embed: (B, T, 21, 4) -> (B, T, 21, num_hid)
        x = self.embed(x)
        
        # Rest remains the same
        ...
""")

# ============================================================================
# 4. TRAINING
# ============================================================================
print("\n\n🎓 BƯỚC 3: TRAINING")
print("-" * 80)

print("""
Không cần thay đổi training loop!

# In train.py

for epoch in range(epochs):
    for batch in dataloader:
        # Dataloader returns augmented data (4 channels)
        l_hands, r_hands, bodies, labels = batch
        
        # l_hands: (B, T, 21, 4)  ← 4 channels instead of 2
        # r_hands: (B, T, 21, 4)
        
        outputs = model(l_hands, r_hands, bodies, training=True)
        loss = criterion(outputs, labels)
        
        # Standard training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Model tự động học:
    - Khi nào dùng shape features (channels 0-1)
    - Khi nào dùng position features (channels 2-3)
    - Hoặc combine cả 2!
""")

# ============================================================================
# 5. DATASET MODIFICATION
# ============================================================================
print("\n\n📁 BƯỚC 4: MODIFY DATASET")
print("-" * 80)

print("""
# In datasets/czech_slr_dataset.py

class CzechSLRDataset(torch_data.Dataset):
    def __init__(self, dataset_filename, use_position=True, ...):
        self.use_position = use_position
        # Load data as usual
        ...
        
    def __getitem__(self, idx):
        # Original code: Get shape-normalized data
        l_hand_shape = ...  # (T, 21, 2)
        r_hand_shape = ...  # (T, 21, 2)
        
        if self.use_position:
            # NEW: Load raw data để tính position
            raw_data = self.load_raw_landmarks(idx)
            
            # Compute position features
            left_pos, right_pos = compute_position_features(raw_data)
            
            # Augment
            l_hand = augment_landmarks_with_position(l_hand_shape, left_pos)
            r_hand = augment_landmarks_with_position(r_hand_shape, right_pos)
        else:
            # Fallback to original (backward compatible)
            l_hand = l_hand_shape
            r_hand = r_hand_shape
        
        return l_hand, r_hand, body, label
""")

# ============================================================================
# 6. EXPECTED IMPROVEMENTS
# ============================================================================
print("\n\n📈 BƯỚC 5: KẾT QUẢ KỲ VỌNG")
print("-" * 80)

print("""
1. GESTURES VỊ TRÍ-DEPENDENT:
   Ví dụ: "Tay trên đầu" vs "Tay trước ngực"
   
   Before (shape only):
   - Model khó phân biệt nếu hand shape giống nhau
   
   After (shape + position):
   - Model học được: Position channel 2-3 khác nhau
   - Accuracy improvement: ~10-15%

2. ROBUST VỚI VARIATION:
   
   Before:
   - Model có thể overfit vào vị trí trong training set
   
   After:
   - Model học: "Shape là primary, position là secondary"
   - Better generalization: ~5-10%

3. INTERPRETABILITY:
   
   - Có thể analyze: Model dùng shape hay position nhiều hơn?
   - Attention weights trên channels 0-1 vs 2-3
   - Ablation study: Mask position channels → see performance drop

4. MINIMAL OVERHEAD:
   
   - Chỉ thêm 2 channels
   - Computation tăng không đáng kể
   - Compatible với existing architecture
""")

# ============================================================================
# 7. ABLATION STUDY
# ============================================================================
print("\n\n🔬 BƯỚC 6: ABLATION STUDY (Để verify)")
print("-" * 80)

print("""
Test 3 variants:

Variant 1: Shape only (Original Siformer)
    Input: (B, T, 21, 2) - x, y only
    → Baseline performance

Variant 2: Position only (!!)
    Input: (B, T, 21, 2) - rel_x, rel_y only
    → Kiểm tra: Position có useful không?

Variant 3: Shape + Position (Hybrid)
    Input: (B, T, 21, 4) - x, y, rel_x, rel_y
    → Expected: Best performance

Expected results:
    Variant 1 (shape): 85% accuracy
    Variant 2 (position): 60% accuracy  ← Worse, confirming shape is primary
    Variant 3 (hybrid): 90% accuracy     ← Best, confirming synergy!
""")

# ============================================================================
# 8. IMPLEMENTATION CHECKLIST
# ============================================================================
print("\n\n✅ CHECKLIST ĐỂ IMPLEMENT")
print("=" * 80)

checklist = [
    ("[ ] 1. Modify Dataset", "Add compute_position_features() and augment function"),
    ("[ ] 2. Modify Encoder", "Change input_dim from 2 to 4 in embedding layer"),
    ("[ ] 3. Update config", "Add --use_position flag in argparse"),
    ("[ ] 4. Test dataloader", "Print sample batch to verify shape (B,T,21,4)"),
    ("[ ] 5. Train baseline", "Original model for comparison"),
    ("[ ] 6. Train hybrid", "New model with position features"),
    ("[ ] 7. Compare results", "Accuracy, confusion matrix, per-class performance"),
    ("[ ] 8. Ablation study", "Test shape-only, position-only, hybrid"),
    ("[ ] 9. Attention analysis", "Visualize which channels model focuses on"),
    ("[ ] 10. Deploy", "Use best variant in production"),
]

for item, desc in checklist:
    print(f"{item:20s} {desc}")

print("\n" + "="*80)
print("🎯 Đây là approach ĐƠN GIẢN NHẤT để kết hợp shape và position")
print("   Chỉ cần 3 thay đổi nhỏ:")
print("   1. Dataset: Thêm position features")
print("   2. Encoder: input_dim 2→4")
print("   3. Training: Không đổi!")
print("="*80)

print("\n📄 Chi tiết các approach khác xem tại:")
print("   Siformer/HYBRID_NORMALIZATION_PROPOSAL.md")
