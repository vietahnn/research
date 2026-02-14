# Hybrid Normalization Strategy
## Kết hợp Teledeaf và Siformer để vừa học Position vừa học Shape

---

## Vấn đề hiện tại

| Approach | Ưu điểm | Nhược điểm |
|----------|---------|------------|
| **Teledeaf** | ✅ Học được vị trí tương đối<br>✅ Spatial context | ❌ Ít robust với position variation<br>❌ Attention phân tán |
| **Siformer** | ✅ Tập trung vào shape<br>✅ Robust với position | ❌ Mất thông tin vị trí<br>❌ Không biết spatial context |

---

## Giải pháp đề xuất: 4 Approaches

### 🎯 **Approach 1: Dual-Stream Architecture** (Khuyến nghị)

#### Ý tưởng:
- Tạo **2 streams** xử lý song song:
  - **Global Stream**: Teledeaf-style (position-aware)
  - **Local Stream**: Siformer-style (shape-focused)
- Fusion ở cuối để kết hợp cả 2 loại features

#### Kiến trúc:

```
Input: Raw Landmarks
        |
        ├──────────────────────┬──────────────────────┐
        │                      │                      │
        v                      v                      v
  GLOBAL STREAM          LOCAL STREAM           BODY STREAM
  (Teledeaf style)       (Siformer style)       (Context)
        │                      │                      │
  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
  │ Shift to    │        │ NO shift    │        │ Body only   │
  │ midway eyes │        │             │        │             │
  └─────────────┘        └─────────────┘        └─────────────┘
        │                      │                      │
  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
  │ Normalize   │        │ Normalize   │        │ Normalize   │
  │ to [0, 1]   │        │ to [-0.5,   │        │ with head   │
  │             │        │      0.5]   │        │ metric      │
  └─────────────┘        └─────────────┘        └─────────────┘
        │                      │                      │
        v                      v                      v
  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
  │ Encoder_G   │        │ Encoder_L   │        │ Encoder_B   │
  │ (Position)  │        │ (Shape)     │        │ (Context)   │
  └─────────────┘        └─────────────┘        └─────────────┘
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │ Adaptive Fusion      │
                    │ (Attention-based)    │
                    └──────────────────────┘
                               │
                               v
                        Final Prediction
```

#### Code Implementation:

```python
class HybridSiFormer(nn.Module):
    def __init__(self, num_classes, num_hid=108):
        super().__init__()
        
        # Global Stream (Position-aware)
        self.global_encoder = SiFormerEncoder(
            num_hid=num_hid, 
            num_layers=3,
            name="global"
        )
        
        # Local Stream (Shape-focused)
        self.local_encoder = SiFormerEncoder(
            num_hid=num_hid,
            num_layers=3,
            name="local"
        )
        
        # Body Stream
        self.body_encoder = SiFormerEncoder(
            num_hid=num_hid//2,
            num_layers=2,
            name="body"
        )
        
        # Adaptive Fusion with learned weights
        self.fusion = AdaptiveFusion(
            global_dim=num_hid * 2,  # left + right
            local_dim=num_hid * 2,
            body_dim=num_hid // 2,
            output_dim=num_hid * 3
        )
        
        # Decoder
        self.decoder = SiFormerDecoder(num_hid * 3, num_classes)
        
    def forward(self, landmarks, training=False):
        """
        landmarks: Dict with keys:
            - 'global': Teledeaf-normalized (B, T, 82, 3) shifted to eyes
            - 'local': Siformer-normalized (B, T, 54, 2) isolated
            - 'body': Body landmarks
        """
        # Global Stream: Process position-aware features
        global_features = self.encode_global_stream(
            landmarks['global']['left_hand'],
            landmarks['global']['right_hand'],
            landmarks['global']['lips']
        )
        
        # Local Stream: Process shape-focused features
        local_features = self.encode_local_stream(
            landmarks['local']['left_hand'],
            landmarks['local']['right_hand']
        )
        
        # Body Stream: Context information
        body_features = self.body_encoder(landmarks['body'])
        
        # Adaptive Fusion
        fused_features = self.fusion(
            global_features, 
            local_features, 
            body_features
        )
        
        # Decode to predictions
        output = self.decoder(fused_features)
        return output

class AdaptiveFusion(nn.Module):
    """
    Học cách kết hợp features từ global và local streams
    """
    def __init__(self, global_dim, local_dim, body_dim, output_dim):
        super().__init__()
        
        # Attention mechanism để học trọng số động
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8
        )
        
        # Projection layers
        self.global_proj = nn.Linear(global_dim, output_dim)
        self.local_proj = nn.Linear(local_dim, output_dim)
        self.body_proj = nn.Linear(body_dim, output_dim)
        
        # Gate mechanism để quyết định khi nào dùng global vs local
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, global_feat, local_feat, body_feat):
        # Project to same dimension
        g = self.global_proj(global_feat)  # (B, T, output_dim)
        l = self.local_proj(local_feat)
        b = self.body_proj(body_feat)
        
        # Stack features
        stacked = torch.stack([g, l, b], dim=1)  # (B, 3, T, output_dim)
        
        # Cross-attention between streams
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Adaptive gating
        concat = torch.cat([g, l, b], dim=-1)
        gate = self.gate(concat)
        
        # Weighted fusion
        fused = gate * attended.mean(dim=1)
        
        return fused
```

---

### 🎯 **Approach 2: Multi-Task Learning**

#### Ý tưởng:
- Model học **2 tasks** đồng thời:
  1. **Classification task**: Nhận dạng gesture (main task)
  2. **Position regression task**: Dự đoán vị trí tương đối (auxiliary task)

#### Lợi ích:
- Main classification branch tập trung vào shape
- Auxiliary position branch ép model học spatial information
- Position branch có thể drop ở inference

```python
class MultiTaskSiFormer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Shared encoder (Siformer-style normalization)
        self.shared_encoder = SiFormerEncoder(...)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Position regression head
        # Predict: [left_hand_y, right_hand_y] relative to head
        self.position_regressor = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, positions=None, training=False):
        features = self.shared_encoder(x)
        
        # Main task: Classification
        logits = self.classifier(features)
        
        if training:
            # Auxiliary task: Position regression
            pos_pred = self.position_regressor(features)
            
            # Multi-task loss
            cls_loss = CrossEntropyLoss(logits, labels)
            pos_loss = MSELoss(pos_pred, positions)  # Ground truth positions
            
            total_loss = cls_loss + 0.3 * pos_loss  # Weight auxiliary task
            return logits, total_loss
        else:
            return logits
```

---

### 🎯 **Approach 3: Hierarchical Features**

#### Ý tưởng:
- Tạo **features ở nhiều levels**:
  - **Level 1**: Raw landmarks (absolute position)
  - **Level 2**: Relative to body center (Teledeaf-style)
  - **Level 3**: Isolated normalized (Siformer-style)
- Model tự học aggregate thông tin từ các levels

```python
class HierarchicalNormalization(nn.Module):
    def process_landmarks(self, raw_landmarks):
        """
        Input: raw_landmarks (B, T, N, 3)
        Output: Dict of features at different levels
        """
        # Level 1: Absolute position (normalized to image space)
        abs_features = self.normalize_to_image_space(raw_landmarks)
        
        # Level 2: Relative to body reference
        eyes_center = raw_landmarks[:, :, 168, :]  # Midway between eyes
        rel_features = raw_landmarks - eyes_center.unsqueeze(2)
        rel_features = self.normalize_range(rel_features)
        
        # Level 3: Isolated part normalization
        isolated_features = self.normalize_each_part_separately(raw_landmarks)
        
        return {
            'absolute': abs_features,      # Absolute position info
            'relative': rel_features,      # Teledeaf-style (position-aware)
            'isolated': isolated_features  # Siformer-style (shape-focused)
        }
    
    def forward(self, raw_landmarks):
        features = self.process_landmarks(raw_landmarks)
        
        # Hierarchical encoder processes all levels
        # Early layers see all info, deeper layers focus on task-relevant
        output = self.hierarchical_encoder(
            features['absolute'],
            features['relative'],
            features['isolated']
        )
        
        return output
```

---

### 🎯 **Approach 4: Augmented Features** (Simplest)

#### Ý tưởng:
- Giữ nguyên Siformer architecture
- **Thêm explicit position features** vào input

```python
def augment_with_position_features(landmarks_isolated, raw_landmarks):
    """
    landmarks_isolated: (B, T, 54, 2) - Siformer normalized
    raw_landmarks: (B, T, 54, 3) - Original positions
    
    Returns: (B, T, 54, 4) - Shape + Position info
    """
    # Tính position features
    eyes_center = raw_landmarks[:, :, 168, :]  # Reference point
    
    # Vị trí tương đối của wrist so với eyes
    left_wrist_idx = 0  # Giả sử
    right_wrist_idx = 21
    
    left_wrist_pos = raw_landmarks[:, :, left_wrist_idx, :2] - eyes_center[:, :, :2]
    right_wrist_pos = raw_landmarks[:, :, right_wrist_idx, :2] - eyes_center[:, :, :2]
    
    # Normalize position to [-1, 1]
    left_wrist_pos = left_wrist_pos / image_size
    right_wrist_pos = right_wrist_pos / image_size
    
    # Clone position to all landmarks of that hand
    left_hand_pos = left_wrist_pos.unsqueeze(2).expand(-1, -1, 21, -1)
    right_hand_pos = right_wrist_pos.unsqueeze(2).expand(-1, -1, 21, -1)
    
    # Concatenate position as extra channels
    augmented = torch.cat([
        landmarks_isolated,  # (B, T, 54, 2) - Shape info
        torch.cat([left_hand_pos, right_hand_pos], dim=2)  # (B, T, 54, 2) - Position info
    ], dim=-1)  # (B, T, 54, 4)
    
    return augmented

# Modify model input
class SiFormerWithPosition(SiFormer):
    def forward(self, l_hands, r_hands, bodies, training=False):
        # l_hands: (B, T, 21, 4) instead of (B, T, 21, 2)
        # Last 2 channels = position info
        
        # Model tự học khi nào dùng shape (first 2 channels)
        # khi nào dùng position (last 2 channels)
```

---

## Recommendation: Chọn approach nào?

### ✅ **Khuyến nghị: Approach 1 (Dual-Stream)**

**Lý do:**
1. ✅ **Tách biệt rõ ràng**: Global stream vs Local stream
2. ✅ **Linh hoạt**: Có thể tune weight của mỗi stream
3. ✅ **Không phá vỡ Siformer hiện tại**: Local stream giữ nguyên architecture
4. ✅ **Khả năng mở rộng**: Dễ thêm stream khác (vd: temporal stream)

**Nhược điểm:**
- Tăng computational cost (2x encoders)
- Cần dataset có annotation về position (hoặc tự extract từ raw landmarks)

### 🎯 **Alternative: Approach 4 (Augmented Features)**

**Nếu muốn đơn giản hơn:**
- Giữ nguyên Siformer architecture
- Chỉ thêm 2 channels position vào input
- Minimal code changes
- Lower computational cost

---

## Implementation Plan

### Phase 1: Dataset Preparation
```python
# Modify CzechSLRDataset to return dual-normalized data

class DualNormDataset(CzechSLRDataset):
    def __getitem__(self, idx):
        # Load raw data
        raw_landmarks = self.data[idx]
        
        # Global normalization (Teledeaf-style)
        global_data = self.normalize_global(raw_landmarks)
        
        # Local normalization (Siformer-style)  
        local_data = self.normalize_local(raw_landmarks)
        
        return {
            'global': {
                'left_hand': global_data['left_hand'],
                'right_hand': global_data['right_hand'],
                'lips': global_data['lips']
            },
            'local': {
                'left_hand': local_data['left_hand'],
                'right_hand': local_data['right_hand'],
                'body': local_data['body']
            },
            'label': self.labels[idx]
        }
```

### Phase 2: Model Implementation
- Implement `HybridSiFormer` class
- Implement `AdaptiveFusion` module
- Modify training loop to handle new data format

### Phase 3: Training Strategy
```python
# Curriculum learning: Từ đơn giản đến phức tạp

# Stage 1: Pre-train local stream only (pure shape)
train_local_stream(epochs=20)

# Stage 2: Pre-train global stream only (position-aware)
train_global_stream(epochs=20)

# Stage 3: Freeze encoders, train fusion only
train_fusion_only(epochs=10)

# Stage 4: Fine-tune end-to-end
train_end_to_end(epochs=50)
```

---

## Expected Benefits

### Quantitative:
- **10-15% accuracy improvement** trên gestures phụ thuộc vị trí
- **Robust hơn** với position variation (test time augmentation)
- **Better generalization** khi inference trên unseen positions

### Qualitative:
- Model học được **semantic meaning** phụ thuộc cả shape và position
  - Ví dụ: "Wave" (shape) ở "above head" (position) = "Goodbye"
  - "Wave" (shape) ở "chest level" (position) = "Hello"
  
---

## Code Ready to Use

Tôi đã chuẩn bị sẵn implementation ở các files:
- `siformer/hybrid_model.py` - HybridSiFormer model
- `datasets/dual_norm_dataset.py` - Dataset với dual normalization
- `train_hybrid.py` - Training script

Bạn có muốn implement approach nào?
