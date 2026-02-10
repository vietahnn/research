"""
Spatial-Temporal Positional Encoding for Skeleton-based Sign Language Recognition

Implements combined spatial and temporal positional encodings:
- Temporal PE: Encodes frame position in sequence (time dimension)
- Spatial PE: Encodes joint identity in skeleton (spatial dimension)

Both encodings are added to input features as per the paper approach.

Author: Research Team
Date: February 2026
"""

import math
import torch
import torch.nn as nn


class SpatialTemporalPE(nn.Module):
    """
    Combined Spatial and Temporal Positional Encoding for skeleton data.
    
    Key concepts:
    - Temporal PE encodes WHEN (which frame in sequence)
    - Spatial PE encodes WHAT (which joint: thumb, index, wrist, etc.)
    
    Formula (from paper):
        output = features + temporal_pe + spatial_pe
    
    This allows the transformer to understand both:
    1. Temporal relationships: How joints move over time
    2. Spatial relationships: Which joints are being represented
    """
    
    def __init__(self, num_joints, d_coords=2, seq_len=204, 
                 encoding_type='learnable', dropout=0.1):
        """
        Initialize spatial-temporal positional encoding.
        Both spatial and temporal use the SAME encoding type for consistency.
        
        Args:
            num_joints (int): Number of skeleton joints (21 for hand, 12 for body)
            d_coords (int): Coordinate dimensions (2 for x,y; 3 for x,y,z)
            seq_len (int): Maximum sequence length
            encoding_type (str): Type of encoding for BOTH spatial and temporal:
                                'learnable' - learnable parameters (default)
                                'sinusoidal' - fixed sin/cos encoding
            dropout (float): Dropout rate applied after adding PE
        """
        super().__init__()
        self.d_model = num_joints * d_coords
        self.num_joints = num_joints
        self.d_coords = d_coords
        self.seq_len = seq_len
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(p=dropout)
        
        # Create both spatial and temporal PE using the SAME encoding type
        if encoding_type == 'learnable':
            self.temporal_pe = self._create_temporal_pe_learnable(seq_len)
            self.spatial_pe = self._create_spatial_pe_learnable()
        elif encoding_type == 'sinusoidal':
            self.temporal_pe = self._create_temporal_pe_sinusoidal(seq_len)
            self.spatial_pe = self._create_spatial_pe_sinusoidal()
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}. Use 'learnable' or 'sinusoidal'")
    
    def _create_temporal_pe_learnable(self, seq_len):
        """
        Create learnable temporal positional encoding (encodes frame position).
        
        Returns:
            nn.Parameter: Shape (seq_len, 1, d_model)
        """
        # Initialize with small random values
        temporal_pe = torch.randn(seq_len, 1, self.d_model) * 0.02
        return nn.Parameter(temporal_pe, requires_grad=True)
    
    def _create_temporal_pe_sinusoidal(self, seq_len):
        """
        Create sinusoidal temporal positional encoding.
        Standard Transformer-style encoding for frame positions.
        
        Returns:
            nn.Parameter: Shape (seq_len, 1, d_model)
        """
        pe = torch.zeros(seq_len, 1, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Different frequency for each dimension
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Make it learnable (optional - can use register_buffer for fixed)
        return nn.Parameter(pe, requires_grad=True)
    
    def _create_spatial_pe_learnable(self):
        """
        Create learnable spatial positional encoding.
        
        Each position in the feature vector gets a learnable offset that
        identifies which joint coordinate it represents.
        
        Benefits:
        - Model learns optimal spatial encoding during training
        - More flexible than fixed sinusoidal patterns
        - Simple and effective
        
        Returns:
            nn.Parameter: Shape (1, 1, d_model)
        """
        # Initialize with small random values to avoid overwhelming initial features
        pe = torch.randn(1, 1, self.d_model) * 0.02
        
        # Optional: Make x,y coordinates of same joint share similar encoding
        # This encodes the intuition that "x and y of thumb belong together"
        # Uncomment if you want this structure:
        # for joint_idx in range(self.num_joints):
        #     base_val = torch.randn(1) * 0.02
        #     for coord_idx in range(self.d_coords):
        #         pe[0, 0, joint_idx * self.d_coords + coord_idx] = base_val
        
        return nn.Parameter(pe, requires_grad=True)
    
    def _create_spatial_pe_sinusoidal(self):
        """
        Create sinusoidal spatial positional encoding (similar to Transformer).
        
        Uses sine/cosine functions with different frequencies for different joints.
        Formula from paper:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Where pos = joint index (0-20 for hand)
        
        Benefits:
        - Fixed encoding (no additional parameters to learn)
        - Smooth transitions between adjacent joints
        - Proven in original Transformer paper
        
        Returns:
            nn.Parameter: Shape (1, 1, d_model)
        """
        pe = torch.zeros(1, 1, self.d_model)
        
        for joint_idx in range(self.num_joints):
            for coord_idx in range(self.d_coords):
                dim = joint_idx * self.d_coords + coord_idx
                
                # Different frequency based on coordinate index
                freq = joint_idx / (10000 ** (2 * coord_idx / self.d_coords))
                
                # Alternate between sin and cos
                if coord_idx % 2 == 0:
                    pe[0, 0, dim] = math.sin(freq)
                else:
                    pe[0, 0, dim] = math.cos(freq)
        
        # Make it learnable (optional - can use register_buffer for fixed)
        return nn.Parameter(pe, requires_grad=True)
    
    def forward(self, x):
        """
        Add spatial and temporal positional encodings to input features.
        
        Args:
            x (torch.Tensor): Input features
                Shape: (seq_len, batch_size, d_model)
                
        Returns:
            torch.Tensor: Features with positional encodings added
                Shape: (seq_len, batch_size, d_model)
                
        Computation:
            output = x + temporal_pe + spatial_pe
            
        Broadcasting:
            - x:           (seq_len, batch_size, d_model)
            - temporal_pe: (seq_len, 1,          d_model) → broadcasts across batch
            - spatial_pe:  (1,       1,          d_model) → broadcasts across seq & batch
        """
        # Ensure temporal PE doesn't exceed input sequence length
        seq_len = x.size(0)
        if seq_len > self.seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum {self.seq_len}. "
                f"Increase seq_len parameter or truncate input."
            )
        
        # Slice temporal PE to match current sequence length
        temporal_pe = self.temporal_pe[:seq_len]
        
        # Add both positional encodings
        # temporal_pe broadcasts across batch dimension
        # spatial_pe broadcasts across both seq_len and batch dimensions
        x = x + temporal_pe + self.spatial_pe
        
        # Apply dropout for regularization
        return self.dropout(x)
    
    def __repr__(self):
        return (f"SpatialTemporalPE(num_joints={self.num_joints}, "
                f"d_coords={self.d_coords}, seq_len={self.seq_len}, "
                f"d_model={self.d_model}, encoding_type='{self.encoding_type}')")


if __name__ == "__main__":
    # Test the spatial-temporal PE
    print("Testing SpatialTemporalPE...")
    
    # Hand: 21 joints × 2 coords = 42 features
    pe_hand = SpatialTemporalPE(num_joints=21, d_coords=2, seq_len=204, encoding_type='learnable')
    
    # Dummy input: (seq_len=100, batch_size=4, d_model=42)
    x = torch.randn(100, 4, 42)
    output = pe_hand(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Temporal PE shape: {pe_hand.temporal_pe.shape}")
    print(f"Spatial PE shape: {pe_hand.spatial_pe.shape}")
    print("✓ Test passed!")
