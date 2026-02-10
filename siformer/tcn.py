"""
Temporal Convolutional Networks (TCN) for Sign Language Recognition

Implementation based on:
- Dilated convolutions with exponentially increasing dilation rates
- Residual connections for gradient flow
- Causal padding to preserve temporal causality
- Weight normalization for training stability

Architecture designed for feature-isolated temporal modeling in Siformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolutions and residual connections.
    
    Architecture:
        Input → Conv1(dilated) → ReLU → Dropout → Conv2(dilated) → ReLU → Dropout
          ↓                                                                    ↓
          └──────────────────── Residual (1x1 conv if needed) ────────────────┘
    
    Args:
        n_inputs: Number of input channels
        n_outputs: Number of output channels
        kernel_size: Size of convolutional kernel (typically 3 or 5)
        dilation: Dilation rate for temporal convolution
        dropout: Dropout probability
        use_weight_norm: Whether to apply weight normalization
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.1, use_weight_norm=True):
        super(TemporalBlock, self).__init__()
        
        # Causal padding: pad only on the left side to preserve causality
        # Total padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        # First dilated convolution
        conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.conv1 = weight_norm(conv1) if use_weight_norm else conv1
        
        # Second dilated convolution
        conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.conv2 = weight_norm(conv2) if use_weight_norm else conv2
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection: 1x1 conv if input/output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using normal distribution"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, channels, seq_len]
        
        Returns:
            Output tensor of shape [batch, channels, seq_len]
        """
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Chop padding to maintain causality
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding]  # Chop padding
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Match sequence length (in case of mismatch due to padding)
        if res.size(2) != out.size(2):
            res = res[:, :, :out.size(2)]
        
        return self.relu(out + res)


class MultiScaleTCN(nn.Module):
    """
    Multi-scale Temporal Convolutional Network with exponentially increasing dilations.
    
    Stack of TemporalBlocks with dilation rates: [1, 2, 4, 8, 16, 32, ...]
    This creates a large receptive field while maintaining parameter efficiency.
    
    Receptive field: r = 1 + 2 * sum(dilations) = 1 + 2 * (2^num_layers - 1)
    Example: 6 layers → r = 1 + 2*(63) = 127 frames
    
    Args:
        num_inputs: Number of input channels
        num_channels: List of output channels for each layer
                     Example: [64, 64, 128, 128, original_dim]
        kernel_size: Kernel size for convolutions (default: 3)
        dropout: Dropout probability (default: 0.1)
        use_weight_norm: Whether to use weight normalization (default: True)
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1, use_weight_norm=True):
        super(MultiScaleTCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, ...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels,
                    kernel_size, dilation_size,
                    dropout, use_weight_norm
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # Calculate and store receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, num_levels)
    
    def _calculate_receptive_field(self, kernel_size, num_layers):
        """
        Calculate theoretical receptive field.
        
        Formula: r = 1 + (kernel_size - 1) * sum(dilations)
        For exponential dilations: sum(2^i for i in 0..L-1) = 2^L - 1
        """
        sum_dilations = sum(2**i for i in range(num_layers))
        receptive_field = 1 + (kernel_size - 1) * 2 * sum_dilations
        return receptive_field
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, channels, seq_len]
        
        Returns:
            Output tensor of shape [batch, channels, seq_len]
        """
        return self.network(x)
    
    def get_receptive_field(self):
        """Return the theoretical receptive field of this TCN"""
        return self.receptive_field


class FeatureIsolatedTCN(nn.Module):
    """
    Feature-isolated TCN modules for Siformer architecture.
    
    Maintains separate TCN for each body part (left hand, right hand, body)
    to preserve the feature isolation design of Siformer.
    
    Args:
        l_hand_dim: Dimension of left hand features (default: 42)
        r_hand_dim: Dimension of right hand features (default: 42)
        body_dim: Dimension of body features (default: 24)
        num_layers: Number of TCN layers (default: 4)
        hidden_dim_factor: Factor to expand hidden dimensions (default: 1.5)
        kernel_size: Kernel size for convolutions (default: 3)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, l_hand_dim=42, r_hand_dim=42, body_dim=24,
                 num_layers=4, hidden_dim_factor=1.5, kernel_size=3, dropout=0.1):
        super(FeatureIsolatedTCN, self).__init__()
        
        # Calculate hidden dimensions (expand then restore)
        l_hand_hidden = int(l_hand_dim * hidden_dim_factor)
        r_hand_hidden = int(r_hand_dim * hidden_dim_factor)
        body_hidden = int(body_dim * hidden_dim_factor)
        
        # Build channel lists for each stream
        # Example for 4 layers: [42] → [64, 64, 64, 42]
        l_hand_channels = self._build_channel_list(l_hand_dim, l_hand_hidden, num_layers)
        r_hand_channels = self._build_channel_list(r_hand_dim, r_hand_hidden, num_layers)
        body_channels = self._build_channel_list(body_dim, body_hidden, num_layers)
        
        # Create TCN for each body part
        self.l_hand_tcn = MultiScaleTCN(
            l_hand_dim, l_hand_channels,
            kernel_size=kernel_size, dropout=dropout
        )
        
        self.r_hand_tcn = MultiScaleTCN(
            r_hand_dim, r_hand_channels,
            kernel_size=kernel_size, dropout=dropout
        )
        
        self.body_tcn = MultiScaleTCN(
            body_dim, body_channels,
            kernel_size=kernel_size, dropout=dropout
        )
        
        print(f"✓ TCN initialized: {num_layers} layers, kernel_size={kernel_size}")
        print(f"  L_hand TCN: {l_hand_dim}D → {l_hand_channels} → {l_hand_dim}D")
        print(f"  R_hand TCN: {r_hand_dim}D → {r_hand_channels} → {r_hand_dim}D")
        print(f"  Body TCN:   {body_dim}D → {body_channels} → {body_dim}D")
        print(f"  Receptive field: {self.l_hand_tcn.get_receptive_field()} frames")
    
    def _build_channel_list(self, input_dim, hidden_dim, num_layers):
        """
        Build channel progression for TCN layers.
        Expand to hidden_dim in middle layers, restore to input_dim at end.
        """
        if num_layers == 1:
            return [input_dim]
        elif num_layers == 2:
            return [hidden_dim, input_dim]
        else:
            # First layer: expand, middle layers: maintain, last layer: restore
            channels = [hidden_dim] * (num_layers - 1) + [input_dim]
            return channels
    
    def forward(self, l_hand, r_hand, body):
        """
        Args:
            l_hand: Left hand features [batch, seq_len, l_hand_dim]
            r_hand: Right hand features [batch, seq_len, r_hand_dim]
            body: Body features [batch, seq_len, body_dim]
        
        Returns:
            Tuple of (l_hand_out, r_hand_out, body_out) with same shapes as inputs
        """
        # TCN expects [batch, channels, seq_len]
        # Transpose from [batch, seq_len, features] → [batch, features, seq_len]
        l_hand_t = l_hand.transpose(1, 2)
        r_hand_t = r_hand.transpose(1, 2)
        body_t = body.transpose(1, 2)
        
        # Apply TCN
        l_hand_tcn = self.l_hand_tcn(l_hand_t)
        r_hand_tcn = self.r_hand_tcn(r_hand_t)
        body_tcn = self.body_tcn(body_t)
        
        # Transpose back to [batch, seq_len, features]
        l_hand_out = l_hand_tcn.transpose(1, 2)
        r_hand_out = r_hand_tcn.transpose(1, 2)
        body_out = body_tcn.transpose(1, 2)
        
        return l_hand_out, r_hand_out, body_out


if __name__ == "__main__":
    """Test TCN modules"""
    print("=" * 60)
    print("Testing TCN Modules")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 100
    
    # Test TemporalBlock
    print("\n1. Testing TemporalBlock:")
    block = TemporalBlock(n_inputs=42, n_outputs=64, kernel_size=3, dilation=4, dropout=0.1)
    x = torch.randn(batch_size, 42, seq_len)
    out = block(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    assert out.shape == (batch_size, 64, seq_len), "Output shape mismatch!"
    print("   ✓ TemporalBlock test passed")
    
    # Test MultiScaleTCN
    print("\n2. Testing MultiScaleTCN:")
    tcn = MultiScaleTCN(num_inputs=42, num_channels=[64, 64, 64, 42], kernel_size=3, dropout=0.1)
    x = torch.randn(batch_size, 42, seq_len)
    out = tcn(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Receptive field: {tcn.get_receptive_field()} frames")
    assert out.shape == (batch_size, 42, seq_len), "Output shape mismatch!"
    print("   ✓ MultiScaleTCN test passed")
    
    # Test FeatureIsolatedTCN
    print("\n3. Testing FeatureIsolatedTCN:")
    fi_tcn = FeatureIsolatedTCN(
        l_hand_dim=42, r_hand_dim=42, body_dim=24,
        num_layers=4, kernel_size=3, dropout=0.1
    )
    l_hand = torch.randn(batch_size, seq_len, 42)
    r_hand = torch.randn(batch_size, seq_len, 42)
    body = torch.randn(batch_size, seq_len, 24)
    
    l_out, r_out, b_out = fi_tcn(l_hand, r_hand, body)
    print(f"   L_hand: {l_hand.shape} → {l_out.shape}")
    print(f"   R_hand: {r_hand.shape} → {r_out.shape}")
    print(f"   Body:   {body.shape} → {b_out.shape}")
    assert l_out.shape == l_hand.shape, "L_hand shape mismatch!"
    assert r_out.shape == r_hand.shape, "R_hand shape mismatch!"
    assert b_out.shape == body.shape, "Body shape mismatch!"
    print("   ✓ FeatureIsolatedTCN test passed")
    
    print("\n" + "=" * 60)
    print("All TCN tests passed! ✓")
    print("=" * 60)
