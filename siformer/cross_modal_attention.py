"""
Bi-directional Multi-Head Cross-Modal Attention Module for Sign Language Recognition

This module implements parallel bi-directional cross-attention between different body parts
(left hand, right hand, and body) with gating mechanisms for adaptive fusion.

Author: Research Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionFusion(nn.Module):
    """
    Parallel bi-directional cross-attention between all body parts.
    Each part can attend to others simultaneously with learnable gating.
    
    This implementation enables:
    - Left hand <-> Right hand interaction
    - Left hand <-> Body interaction  
    - Right hand <-> Body interaction
    - Bi-directional attention (all parts learn from each other)
    - Gated fusion to control the amount of cross-modal information
    """
    
    def __init__(self, d_lhand, d_rhand, d_body, num_heads=3, dropout=0.1):
        """
        Initialize the cross-modal attention fusion module.
        
        Args:
            d_lhand (int): Dimension of left hand features
            d_rhand (int): Dimension of right hand features
            d_body (int): Dimension of body features
            num_heads (int): Number of attention heads (default: 3)
            dropout (float): Dropout probability (default: 0.1)
        """
        super(CrossModalAttentionFusion, self).__init__()
        
        # Auto-adjust num_heads to ensure divisibility
        original_num_heads = num_heads
        while num_heads > 0:
            if d_lhand % num_heads == 0 and d_rhand % num_heads == 0 and d_body % num_heads == 0:
                break
            num_heads -= 1
        
        if num_heads == 0:
            num_heads = 1  # Fallback to 1 head
        
        if num_heads != original_num_heads:
            print(f"Warning: Adjusted cross-attention heads from {original_num_heads} to {num_heads}")
            print(f"  to be divisible by dimensions: lhand={d_lhand}, rhand={d_rhand}, body={d_body}")
        
        self.d_lhand = d_lhand
        self.d_rhand = d_rhand
        self.d_body = d_body
        self.num_heads = num_heads
        
        # ===== Left Hand Attention Modules =====
        # Left hand attending to right hand and body
        self.lh2rh_attn = nn.MultiheadAttention(
            d_lhand, num_heads, dropout=dropout, batch_first=True
        )
        self.lh2body_attn = nn.MultiheadAttention(
            d_lhand, num_heads, dropout=dropout, batch_first=True
        )
        
        # ===== Right Hand Attention Modules =====
        # Right hand attending to left hand and body
        self.rh2lh_attn = nn.MultiheadAttention(
            d_rhand, num_heads, dropout=dropout, batch_first=True
        )
        self.rh2body_attn = nn.MultiheadAttention(
            d_rhand, num_heads, dropout=dropout, batch_first=True
        )
        
        # ===== Body Attention Modules =====
        # Body attending to left hand and right hand
        self.body2lh_attn = nn.MultiheadAttention(
            d_body, num_heads, dropout=dropout, batch_first=True
        )
        self.body2rh_attn = nn.MultiheadAttention(
            d_body, num_heads, dropout=dropout, batch_first=True
        )
        
        # ===== Projection Layers =====
        # Project features to match query dimensions for cross-attention
        
        # Projections for left hand queries
        self.rh_to_lh_proj = nn.Linear(d_rhand, d_lhand)
        self.body_to_lh_proj = nn.Linear(d_body, d_lhand)
        
        # Projections for right hand queries
        self.lh_to_rh_proj = nn.Linear(d_lhand, d_rhand)
        self.body_to_rh_proj = nn.Linear(d_body, d_rhand)
        
        # Projections for body queries
        self.lh_to_body_proj = nn.Linear(d_lhand, d_body)
        self.rh_to_body_proj = nn.Linear(d_rhand, d_body)
        
        # ===== Gating Mechanisms =====
        # Learnable gates to control how much cross-modal information to incorporate
        self.lh_gate = nn.Sequential(
            nn.Linear(d_lhand * 3, d_lhand),
            nn.Sigmoid()
        )
        self.rh_gate = nn.Sequential(
            nn.Linear(d_rhand * 3, d_rhand),
            nn.Sigmoid()
        )
        self.body_gate = nn.Sequential(
            nn.Linear(d_body * 3, d_body),
            nn.Sigmoid()
        )
        
        # ===== Layer Normalization =====
        self.lh_norm = nn.LayerNorm(d_lhand)
        self.rh_norm = nn.LayerNorm(d_rhand)
        self.body_norm = nn.LayerNorm(d_body)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lh_feat, rh_feat, body_feat):
        """
        Forward pass of cross-modal attention fusion.
        
        Args:
            lh_feat: Left hand features [Batch, Length, d_lhand] or [Length, Batch, d_lhand]
            rh_feat: Right hand features [Batch, Length, d_rhand] or [Length, Batch, d_rhand]
            body_feat: Body features [Batch, Length, d_body] or [Length, Batch, d_body]
            
        Returns:
            Tuple of (lh_out, rh_out, body_out) with enhanced cross-modal features
        """
        # Handle both batch_first=True and batch_first=False
        # Assuming input is [Length, Batch, Dim], convert to [Batch, Length, Dim]
        if lh_feat.dim() == 3 and lh_feat.size(1) < lh_feat.size(0):
            # Likely [L, B, D] format, transpose to [B, L, D]
            lh_feat = lh_feat.transpose(0, 1)
            rh_feat = rh_feat.transpose(0, 1)
            body_feat = body_feat.transpose(0, 1)
            needs_transpose_back = True
        else:
            needs_transpose_back = False
        
        # ===== Left Hand Cross-Attention =====
        # Project right hand and body to left hand dimension
        rh_proj = self.rh_to_lh_proj(rh_feat)  # [B, L, d_lhand]
        body_proj = self.body_to_lh_proj(body_feat)  # [B, L, d_lhand]
        
        # Left hand attends to right hand
        lh_from_rh, _ = self.lh2rh_attn(lh_feat, rh_proj, rh_proj)
        lh_from_rh = self.dropout(lh_from_rh)
        
        # Left hand attends to body
        lh_from_body, _ = self.lh2body_attn(lh_feat, body_proj, body_proj)
        lh_from_body = self.dropout(lh_from_body)
        
        # Gated fusion for left hand
        lh_concat = torch.cat([lh_feat, lh_from_rh, lh_from_body], dim=-1)
        lh_gate_weights = self.lh_gate(lh_concat)
        lh_fused = lh_feat + lh_gate_weights * (lh_from_rh + lh_from_body)
        lh_out = self.lh_norm(lh_fused)
        
        # ===== Right Hand Cross-Attention =====
        # Project left hand and body to right hand dimension
        lh_proj = self.lh_to_rh_proj(lh_feat)  # [B, L, d_rhand]
        body_proj_r = self.body_to_rh_proj(body_feat)  # [B, L, d_rhand]
        
        # Right hand attends to left hand
        rh_from_lh, _ = self.rh2lh_attn(rh_feat, lh_proj, lh_proj)
        rh_from_lh = self.dropout(rh_from_lh)
        
        # Right hand attends to body
        rh_from_body, _ = self.rh2body_attn(rh_feat, body_proj_r, body_proj_r)
        rh_from_body = self.dropout(rh_from_body)
        
        # Gated fusion for right hand
        rh_concat = torch.cat([rh_feat, rh_from_lh, rh_from_body], dim=-1)
        rh_gate_weights = self.rh_gate(rh_concat)
        rh_fused = rh_feat + rh_gate_weights * (rh_from_lh + rh_from_body)
        rh_out = self.rh_norm(rh_fused)
        
        # ===== Body Cross-Attention =====
        # Project left hand and right hand to body dimension
        lh_proj_b = self.lh_to_body_proj(lh_feat)  # [B, L, d_body]
        rh_proj_b = self.rh_to_body_proj(rh_feat)  # [B, L, d_body]
        
        # Body attends to left hand
        body_from_lh, _ = self.body2lh_attn(body_feat, lh_proj_b, lh_proj_b)
        body_from_lh = self.dropout(body_from_lh)
        
        # Body attends to right hand
        body_from_rh, _ = self.body2rh_attn(body_feat, rh_proj_b, rh_proj_b)
        body_from_rh = self.dropout(body_from_rh)
        
        # Gated fusion for body
        body_concat = torch.cat([body_feat, body_from_lh, body_from_rh], dim=-1)
        body_gate_weights = self.body_gate(body_concat)
        body_fused = body_feat + body_gate_weights * (body_from_lh + body_from_rh)
        body_out = self.body_norm(body_fused)
        
        # Transpose back if needed
        if needs_transpose_back:
            lh_out = lh_out.transpose(0, 1)
            rh_out = rh_out.transpose(0, 1)
            body_out = body_out.transpose(0, 1)
        
        return lh_out, rh_out, body_out


class UniDirectionalCrossModalAttention(nn.Module):
    """
    Uni-directional cross-modal attention module - more efficient than bi-directional.
    
    Supports multiple attention strategies:
    - 'hands_to_body': Hands learn from body context (body provides global context to hands)
    - 'body_to_hands': Body learns from hand details (hands provide gesture details to body)
    - 'hands_bidirectional': Only left-right hand interaction (no body involvement)
    - 'chain': Chain attention - LH→RH→Body→LH (each pair interacts only once)
    - 'three_pairs': Three independent attentions - LH→Body, RH→Body, RH→LH
    
    This reduces parameters and computation by ~50% compared to bi-directional attention.
    """
    
    def __init__(self, d_lhand, d_rhand, d_body, num_heads=2, dropout=0.1, direction='three_pairs'):
        """
        Args:
            d_lhand (int): Left hand feature dimension
            d_rhand (int): Right hand feature dimension
            d_body (int): Body feature dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            direction (str): Attention direction - 'hands_to_body', 'body_to_hands', 'hands_bidirectional', 'chain', or 'three_pairs'
        """
        super(UniDirectionalCrossModalAttention, self).__init__()
        
        # Auto-adjust num_heads for divisibility
        original_num_heads = num_heads
        while num_heads > 0:
            if d_lhand % num_heads == 0 and d_rhand % num_heads == 0 and d_body % num_heads == 0:
                break
            num_heads -= 1
        
        if num_heads == 0:
            num_heads = 1
            
        if num_heads != original_num_heads:
            print(f"Warning: Adjusted cross-attention heads from {original_num_heads} to {num_heads}")
        
        self.direction = direction
        self.d_lhand = d_lhand
        self.d_rhand = d_rhand
        self.d_body = d_body
        self.num_heads = num_heads
        
        print(f"Initializing Uni-directional Cross-Modal Attention: {direction} with {num_heads} heads")
        
        if direction == 'hands_to_body':
            # Hands attend to body (body provides context to hands)
            self.lh_attn = nn.MultiheadAttention(d_lhand, num_heads, dropout=dropout, batch_first=True)
            self.rh_attn = nn.MultiheadAttention(d_rhand, num_heads, dropout=dropout, batch_first=True)
            
            self.body_to_lh_proj = nn.Linear(d_body, d_lhand)
            self.body_to_rh_proj = nn.Linear(d_body, d_rhand)
            
            self.lh_gate = nn.Sequential(nn.Linear(d_lhand * 2, d_lhand), nn.Sigmoid())
            self.rh_gate = nn.Sequential(nn.Linear(d_rhand * 2, d_rhand), nn.Sigmoid())
            
            self.lh_norm = nn.LayerNorm(d_lhand)
            self.rh_norm = nn.LayerNorm(d_rhand)
            
        elif direction == 'body_to_hands':
            # Body attends to both hands (learns from hand gestures)
            self.body_from_lh_attn = nn.MultiheadAttention(d_body, num_heads, dropout=dropout, batch_first=True)
            self.body_from_rh_attn = nn.MultiheadAttention(d_body, num_heads, dropout=dropout, batch_first=True)
            
            self.lh_to_body_proj = nn.Linear(d_lhand, d_body)
            self.rh_to_body_proj = nn.Linear(d_rhand, d_body)
            
            self.body_gate = nn.Sequential(nn.Linear(d_body * 3, d_body), nn.Sigmoid())
            self.body_norm = nn.LayerNorm(d_body)
            
        elif direction == 'hands_bidirectional':
            # Only hand-to-hand interaction
            self.lh_to_rh_attn = nn.MultiheadAttention(d_lhand, num_heads, dropout=dropout, batch_first=True)
            self.rh_to_lh_attn = nn.MultiheadAttention(d_rhand, num_heads, dropout=dropout, batch_first=True)
            
            self.rh_to_lh_proj = nn.Linear(d_rhand, d_lhand)
            self.lh_to_rh_proj = nn.Linear(d_lhand, d_rhand)
            
            self.lh_gate = nn.Sequential(nn.Linear(d_lhand * 2, d_lhand), nn.Sigmoid())
            self.rh_gate = nn.Sequential(nn.Linear(d_rhand * 2, d_rhand), nn.Sigmoid())
            
            self.lh_norm = nn.LayerNorm(d_lhand)
            self.rh_norm = nn.LayerNorm(d_rhand)
            
        elif direction == 'chain':
            # Chain attention: LH → RH → Body → LH (each interaction happens only once)
            # Left hand receives from body
            self.lh_attn = nn.MultiheadAttention(d_lhand, num_heads, dropout=dropout, batch_first=True)
            # Right hand receives from left hand
            self.rh_attn = nn.MultiheadAttention(d_rhand, num_heads, dropout=dropout, batch_first=True)
            # Body receives from right hand
            self.body_attn = nn.MultiheadAttention(d_body, num_heads, dropout=dropout, batch_first=True)
            
            # Projection layers
            self.body_to_lh_proj = nn.Linear(d_body, d_lhand)  # For body → lh
            self.lh_to_rh_proj = nn.Linear(d_lhand, d_rhand)    # For lh → rh
            self.rh_to_body_proj = nn.Linear(d_rhand, d_body)  # For rh → body
            
            # Gating mechanisms
            self.lh_gate = nn.Sequential(nn.Linear(d_lhand * 2, d_lhand), nn.Sigmoid())
            self.rh_gate = nn.Sequential(nn.Linear(d_rhand * 2, d_rhand), nn.Sigmoid())
            self.body_gate = nn.Sequential(nn.Linear(d_body * 2, d_body), nn.Sigmoid())
            
            # Layer normalization
            self.lh_norm = nn.LayerNorm(d_lhand)
            self.rh_norm = nn.LayerNorm(d_rhand)
            self.body_norm = nn.LayerNorm(d_body)
            
        elif direction == 'three_pairs':
            # Three independent attentions: LH→Body, RH→Body, RH→LH
            # Attention 1: Left hand attends to body
            self.lh_attn = nn.MultiheadAttention(d_lhand, num_heads, dropout=dropout, batch_first=True)
            # Attention 2: Right hand attends to body
            self.rh_body_attn = nn.MultiheadAttention(d_rhand, num_heads, dropout=dropout, batch_first=True)
            # Attention 3: Right hand attends to left hand
            self.rh_lh_attn = nn.MultiheadAttention(d_rhand, num_heads, dropout=dropout, batch_first=True)
            
            # Projection layers
            self.body_to_lh_proj = nn.Linear(d_body, d_lhand)   # For body → lh
            self.body_to_rh_proj = nn.Linear(d_body, d_rhand)   # For body → rh
            self.lh_to_rh_proj = nn.Linear(d_lhand, d_rhand)    # For lh → rh
            
            # Gating mechanisms (RH receives from 2 sources, so needs special handling)
            self.lh_gate = nn.Sequential(nn.Linear(d_lhand * 2, d_lhand), nn.Sigmoid())
            self.rh_gate = nn.Sequential(nn.Linear(d_rhand * 3, d_rhand), nn.Sigmoid())  # RH gets from body + lh
            
            # Layer normalization
            self.lh_norm = nn.LayerNorm(d_lhand)
            self.rh_norm = nn.LayerNorm(d_rhand)
            
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lh_feat, rh_feat, body_feat):
        """
        Forward pass with uni-directional attention.
        
        Args:
            lh_feat: Left hand features [Length, Batch, d_lhand] or [Batch, Length, d_lhand]
            rh_feat: Right hand features [Length, Batch, d_rhand] or [Batch, Length, d_rhand]
            body_feat: Body features [Length, Batch, d_body] or [Batch, Length, d_body]
            
        Returns:
            Tuple of (lh_out, rh_out, body_out)
        """
        # Handle both [L, B, D] and [B, L, D] formats
        if lh_feat.dim() == 3 and lh_feat.size(1) < lh_feat.size(0):
            lh_feat = lh_feat.transpose(0, 1)
            rh_feat = rh_feat.transpose(0, 1)
            body_feat = body_feat.transpose(0, 1)
            needs_transpose_back = True
        else:
            needs_transpose_back = False
        
        if self.direction == 'hands_to_body':
            # Hands attend to body
            body_proj_l = self.body_to_lh_proj(body_feat)
            body_proj_r = self.body_to_rh_proj(body_feat)
            
            lh_from_body, _ = self.lh_attn(lh_feat, body_proj_l, body_proj_l)
            rh_from_body, _ = self.rh_attn(rh_feat, body_proj_r, body_proj_r)
            
            lh_from_body = self.dropout(lh_from_body)
            rh_from_body = self.dropout(rh_from_body)
            
            # Gated fusion
            lh_gate_weights = self.lh_gate(torch.cat([lh_feat, lh_from_body], dim=-1))
            rh_gate_weights = self.rh_gate(torch.cat([rh_feat, rh_from_body], dim=-1))
            
            lh_out = self.lh_norm(lh_feat + lh_gate_weights * lh_from_body)
            rh_out = self.rh_norm(rh_feat + rh_gate_weights * rh_from_body)
            body_out = body_feat
            
        elif self.direction == 'body_to_hands':
            # Body attends to both hands
            lh_proj = self.lh_to_body_proj(lh_feat)
            rh_proj = self.rh_to_body_proj(rh_feat)
            
            body_from_lh, _ = self.body_from_lh_attn(body_feat, lh_proj, lh_proj)
            body_from_rh, _ = self.body_from_rh_attn(body_feat, rh_proj, rh_proj)
            
            body_from_lh = self.dropout(body_from_lh)
            body_from_rh = self.dropout(body_from_rh)
            
            # Gated fusion
            body_gate_weights = self.body_gate(torch.cat([body_feat, body_from_lh, body_from_rh], dim=-1))
            body_out = self.body_norm(body_feat + body_gate_weights * (body_from_lh + body_from_rh))
            
            lh_out = lh_feat
            rh_out = rh_feat
            
        elif self.direction == 'hands_bidirectional':
            # Hand-to-hand interaction only
            rh_proj = self.rh_to_lh_proj(rh_feat)
            lh_proj = self.lh_to_rh_proj(lh_feat)
            
            lh_from_rh, _ = self.lh_to_rh_attn(lh_feat, rh_proj, rh_proj)
            rh_from_lh, _ = self.rh_to_lh_attn(rh_feat, lh_proj, lh_proj)
            
            lh_from_rh = self.dropout(lh_from_rh)
            rh_from_lh = self.dropout(rh_from_lh)
            
            # Gated fusion
            lh_gate_weights = self.lh_gate(torch.cat([lh_feat, lh_from_rh], dim=-1))
            rh_gate_weights = self.rh_gate(torch.cat([rh_feat, rh_from_lh], dim=-1))
            
            lh_out = self.lh_norm(lh_feat + lh_gate_weights * lh_from_rh)
            rh_out = self.rh_norm(rh_feat + rh_gate_weights * rh_from_lh)
            body_out = body_feat
            
        elif self.direction == 'chain':
            # Chain attention: LH → RH → Body → LH
            # Each modality only receives information from one other modality (truly uni-directional)
            
            # Step 1: Body → LH (body provides context to left hand)
            body_proj = self.body_to_lh_proj(body_feat)
            lh_from_body, _ = self.lh_attn(lh_feat, body_proj, body_proj)
            lh_from_body = self.dropout(lh_from_body)
            
            # Step 2: LH → RH (left hand provides info to right hand)
            lh_proj = self.lh_to_rh_proj(lh_feat)
            rh_from_lh, _ = self.rh_attn(rh_feat, lh_proj, lh_proj)
            rh_from_lh = self.dropout(rh_from_lh)
            
            # Step 3: RH → Body (right hand provides info back to body)
            rh_proj = self.rh_to_body_proj(rh_feat)
            body_from_rh, _ = self.body_attn(body_feat, rh_proj, rh_proj)
            body_from_rh = self.dropout(body_from_rh)
            
            # Gated fusion for all three modalities
            lh_gate_weights = self.lh_gate(torch.cat([lh_feat, lh_from_body], dim=-1))
            rh_gate_weights = self.rh_gate(torch.cat([rh_feat, rh_from_lh], dim=-1))
            body_gate_weights = self.body_gate(torch.cat([body_feat, body_from_rh], dim=-1))
            
            lh_out = self.lh_norm(lh_feat + lh_gate_weights * lh_from_body)
            rh_out = self.rh_norm(rh_feat + rh_gate_weights * rh_from_lh)
            body_out = self.body_norm(body_feat + body_gate_weights * body_from_rh)
            
        elif self.direction == 'three_pairs':
            # Three independent attentions: LH→Body, RH→Body, RH→LH
            
            # Attention 1: Left hand attends to body
            body_proj_lh = self.body_to_lh_proj(body_feat)
            lh_from_body, _ = self.lh_attn(lh_feat, body_proj_lh, body_proj_lh)
            lh_from_body = self.dropout(lh_from_body)
            
            # Attention 2: Right hand attends to body
            body_proj_rh = self.body_to_rh_proj(body_feat)
            rh_from_body, _ = self.rh_body_attn(rh_feat, body_proj_rh, body_proj_rh)
            rh_from_body = self.dropout(rh_from_body)
            
            # Attention 3: Right hand attends to left hand
            lh_proj = self.lh_to_rh_proj(lh_feat)
            rh_from_lh, _ = self.rh_lh_attn(rh_feat, lh_proj, lh_proj)
            rh_from_lh = self.dropout(rh_from_lh)
            
            # Gated fusion
            # Left hand only receives from body
            lh_gate_weights = self.lh_gate(torch.cat([lh_feat, lh_from_body], dim=-1))
            lh_out = self.lh_norm(lh_feat + lh_gate_weights * lh_from_body)
            
            # Right hand receives from both body and left hand
            rh_gate_weights = self.rh_gate(torch.cat([rh_feat, rh_from_body, rh_from_lh], dim=-1))
            rh_out = self.rh_norm(rh_feat + rh_gate_weights * (rh_from_body + rh_from_lh))
            
            # Body unchanged
            body_out = body_feat
        
        # Transpose back if needed
        if needs_transpose_back:
            lh_out = lh_out.transpose(0, 1)
            rh_out = rh_out.transpose(0, 1)
            body_out = body_out.transpose(0, 1)
        
        return lh_out, rh_out, body_out


class SimplifiedCrossModalAttention(nn.Module):
    """
    A simplified version of cross-modal attention for ablation studies.
    Only applies uni-directional attention (hands -> body).
    """
    
    def __init__(self, d_lhand, d_rhand, d_body, num_heads=2, dropout=0.1):
        super(SimplifiedCrossModalAttention, self).__init__()
        
        self.lh2body_attn = nn.MultiheadAttention(
            d_lhand, num_heads, dropout=dropout, batch_first=True
        )
        self.rh2body_attn = nn.MultiheadAttention(
            d_rhand, num_heads, dropout=dropout, batch_first=True
        )
        
        self.body_to_lh_proj = nn.Linear(d_body, d_lhand)
        self.body_to_rh_proj = nn.Linear(d_body, d_rhand)
        
        self.lh_norm = nn.LayerNorm(d_lhand)
        self.rh_norm = nn.LayerNorm(d_rhand)
        
    def forward(self, lh_feat, rh_feat, body_feat):
        """Simplified forward pass with only hands->body attention."""
        # Handle dimension conversion
        if lh_feat.dim() == 3 and lh_feat.size(1) < lh_feat.size(0):
            lh_feat = lh_feat.transpose(0, 1)
            rh_feat = rh_feat.transpose(0, 1)
            body_feat = body_feat.transpose(0, 1)
            needs_transpose_back = True
        else:
            needs_transpose_back = False
        
        # Hands attend to body
        body_proj_l = self.body_to_lh_proj(body_feat)
        body_proj_r = self.body_to_rh_proj(body_feat)
        
        lh_enhanced, _ = self.lh2body_attn(lh_feat, body_proj_l, body_proj_l)
        rh_enhanced, _ = self.rh2body_attn(rh_feat, body_proj_r, body_proj_r)
        
        lh_out = self.lh_norm(lh_feat + lh_enhanced)
        rh_out = self.rh_norm(rh_feat + rh_enhanced)
        
        if needs_transpose_back:
            lh_out = lh_out.transpose(0, 1)
            rh_out = rh_out.transpose(0, 1)
            body_feat = body_feat.transpose(0, 1)
        
        return lh_out, rh_out, body_feat
