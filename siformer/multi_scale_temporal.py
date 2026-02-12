"""
Multi-Scale Temporal Modeling Module for Sign Language Recognition

This module implements parallel temporal convolutions with different kernel sizes
to capture sign patterns at multiple temporal scales (fast, medium, slow motions).

Author: Research Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalModule(nn.Module):
    """
    Multi-scale temporal convolution for capturing varying signing speeds.
    
    Uses parallel convolutional branches with different kernel sizes to capture
    temporal patterns at different scales:
    - Kernel 1: Instantaneous changes (fast motions like snaps, claps)
    - Kernel 3: Short-term patterns (quick gestures)
    - Kernel 5: Medium-term dynamics (regular signing speed)
    - Kernel 7: Long-term context (slow movements, holds)
    
    Args:
        d_model: Feature dimension (e.g., 42 for hands, 24 for body)
        scales: List of kernel sizes for different temporal scales
        reduction: Channel reduction ratio for efficiency (not used if groups>1)
        dropout: Dropout probability for regularization
    
    Example:
        >>> module = MultiScaleTemporalModule(d_model=42, scales=[1, 3, 5, 7])
        >>> x = torch.randn(24, 204, 42)  # [Batch, SeqLen, Features]
        >>> out = module(x)  # [24, 204, 42] - same shape, enhanced features
    """
    
    def __init__(self, d_model, scales=[1, 3, 5, 7], reduction=4, dropout=0.1):
        super(MultiScaleTemporalModule, self).__init__()
        self.scales = scales
        self.d_model = d_model
        
        # Each branch outputs d_model // len(scales) channels for efficiency
        branch_dim = d_model // len(scales)
        
        # Ensure branch_dim is valid
        if branch_dim == 0:
            branch_dim = d_model
            print(f"Warning: d_model={d_model} too small for {len(scales)} scales. Using full dimension per branch.")
        
        # Parallel temporal convolution branches
        self.branches = nn.ModuleList()
        for kernel_size in scales:
            padding = kernel_size // 2  # Keep sequence length unchanged
            branch = nn.Sequential(
                nn.Conv1d(
                    d_model, 
                    branch_dim, 
                    kernel_size=kernel_size, 
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm1d(branch_dim),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Fusion layer to combine multi-scale features
        fusion_input_dim = branch_dim * len(scales)
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_input_dim, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection with learnable gate
        self.gate = nn.Parameter(torch.tensor([0.1]))  # Start with small contribution
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through multi-scale temporal module.
        
        Args:
            x: Input features [Batch, Seq_Length, Feature_Dim] or [Seq_Length, Batch, Feature_Dim]
        
        Returns:
            Enhanced features with multi-scale temporal patterns, same shape as input
        """
        # Handle both input formats (batch_first vs sequence_first)
        needs_transpose = False
        if x.dim() == 3:
            if x.size(0) < x.size(1):  # Likely [SeqLen, Batch, Dim]
                x = x.transpose(0, 1)  # -> [Batch, SeqLen, Dim]
                needs_transpose = True
        
        # Store input for residual connection
        residual = x
        batch_size, seq_len, feat_dim = x.shape
        
        # Conv1D expects [Batch, Channels, Length]
        x = x.transpose(1, 2)  # [B, D, L]
        
        # Process through each temporal scale branch
        multi_scale_features = []
        for branch in self.branches:
            feat = branch(x)  # [B, branch_dim, L]
            multi_scale_features.append(feat)
        
        # Concatenate along channel dimension
        x = torch.cat(multi_scale_features, dim=1)  # [B, branch_dim * num_scales, L]
        
        # Fuse multi-scale features
        x = self.fusion(x)  # [B, D, L]
        
        # Back to [B, L, D]
        x = x.transpose(1, 2)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Residual connection with learnable gating
        # Gate allows model to control how much multi-scale info to use
        x = residual + self.gate * x
        
        # Restore original format if needed
        if needs_transpose:
            x = x.transpose(0, 1)  # [L, B, D]
        
        return x


class TemporalScaleAttention(nn.Module):
    """
    Optional: Attention-based fusion of multi-scale features.
    
    Instead of simple concatenation, learn to weight different scales
    based on their importance for each sample.
    """
    
    def __init__(self, d_model, num_scales):
        super(TemporalScaleAttention, self).__init__()
        self.num_scales = num_scales
        
        # Attention network to compute scale importance
        self.scale_attention = nn.Sequential(
            nn.Linear(d_model * num_scales, num_scales),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, scale_features):
        """
        Args:
            scale_features: List of [Batch, SeqLen, Dim] from different scales
        
        Returns:
            Weighted combination of scale features
        """
        # Stack scales
        stacked = torch.stack(scale_features, dim=-1)  # [B, L, D, num_scales]
        B, L, D, S = stacked.shape
        
        # Compute attention weights (global pooling + attention)
        pooled = stacked.mean(dim=1)  # [B, D, S]
        pooled = pooled.view(B, -1)  # [B, D*S]
        
        weights = self.scale_attention(pooled)  # [B, S]
        weights = weights.view(B, 1, 1, S)  # [B, 1, 1, S]
        
        # Weighted sum
        output = (stacked * weights).sum(dim=-1)  # [B, L, D]
        
        return output
