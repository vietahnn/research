"""
Cross-Modal Consistency Loss (CMCL) for Sign Language Recognition

This module implements a novel loss function that leverages cross-modal attention
patterns to improve sign language recognition by enforcing consistency between
different body parts (left hand, right hand, and body).

The loss combines:
1. Standard Cross-Entropy for classification
2. Consistency regularization between hand attention patterns
3. Feature alignment between body parts

Author: Research Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalConsistencyLoss(nn.Module):
    """
    Cross-Modal Consistency Loss (CMCL) for sign language recognition.
    
    This loss function exploits the structural relationships between different body parts
    in sign language by enforcing consistency in their cross-attention patterns and
    feature representations.
    
    Loss Components:
    - L_CE: Standard cross-entropy classification loss
    - L_consistency: KL divergence between symmetric attention patterns (lh->rh vs rh->lh)
    - L_alignment: Cosine similarity-based alignment between hand and body features
    
    Total Loss: L_CMCL = L_CE + λ₁ * L_consistency + λ₂ * L_alignment
    """
    
    def __init__(self, num_classes, lambda_consistency=0.1, lambda_alignment=0.05, temperature=1.0):
        """
        Initialize CMCL loss.
        
        Args:
            num_classes (int): Number of sign language classes
            lambda_consistency (float): Weight for consistency loss (default: 0.1)
            lambda_alignment (float): Weight for alignment loss (default: 0.05)
            temperature (float): Temperature for softening attention distributions (default: 1.0)
        """
        super(CrossModalConsistencyLoss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_consistency = lambda_consistency
        self.lambda_alignment = lambda_alignment
        self.temperature = temperature
        
        # Standard cross-entropy for classification
        self.ce_loss = nn.CrossEntropyLoss()
        
        # KL divergence for attention consistency
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, outputs, labels, features=None, return_components=False):
        """
        Compute CMCL loss.
        
        Args:
            outputs (torch.Tensor): Model predictions [batch_size, num_classes]
            labels (torch.Tensor): Ground truth labels [batch_size]
            features (dict, optional): Dictionary containing:
                - 'lh_feat': Left hand features [batch_size, seq_len, d_lhand]
                - 'rh_feat': Right hand features [batch_size, seq_len, d_rhand]
                - 'body_feat': Body features [batch_size, seq_len, d_body]
                - 'attn_lh2rh': Attention weights from left hand to right hand
                - 'attn_rh2lh': Attention weights from right hand to left hand
            return_components (bool): If True, return individual loss components
            
        Returns:
            If return_components=False: Total loss (scalar)
            If return_components=True: Dict with 'total', 'ce', 'consistency', 'alignment'
        """
        # Classification loss (always computed)
        loss_ce = self.ce_loss(outputs, labels)
        
        # Initialize auxiliary losses
        loss_consistency = torch.tensor(0.0, device=outputs.device)
        loss_alignment = torch.tensor(0.0, device=outputs.device)
        
        # Compute auxiliary losses if features are provided
        if features is not None:
            # 1. Consistency Loss: Enforce symmetry between hand attention patterns
            if 'attn_lh2rh' in features and 'attn_rh2lh' in features:
                loss_consistency = self._compute_consistency_loss(
                    features['attn_lh2rh'], 
                    features['attn_rh2lh']
                )
            
            # 2. Alignment Loss: Encourage feature alignment between body parts
            if 'lh_feat' in features and 'rh_feat' in features and 'body_feat' in features:
                loss_alignment = self._compute_alignment_loss(
                    features['lh_feat'],
                    features['rh_feat'],
                    features['body_feat']
                )
        
        # Total loss
        total_loss = (
            loss_ce + 
            self.lambda_consistency * loss_consistency + 
            self.lambda_alignment * loss_alignment
        )
        
        if return_components:
            return {
                'total': total_loss,
                'ce': loss_ce,
                'consistency': loss_consistency,
                'alignment': loss_alignment
            }
        
        return total_loss
    
    def _compute_consistency_loss(self, attn_lh2rh, attn_rh2lh):
        """
        Compute consistency loss using KL divergence between symmetric attention patterns.
        
        The intuition is that if left hand attends to right hand, the reverse attention
        should have similar patterns (symmetric interaction).
        
        Args:
            attn_lh2rh: Attention from left hand to right hand 
                        [batch, heads, len, len] or [batch, len, len]
            attn_rh2lh: Attention from right hand to left hand 
                        [batch, heads, len, len] or [batch, len, len]
            
        Returns:
            Consistency loss (scalar)
        """
        if attn_lh2rh is None or attn_rh2lh is None:
            return torch.tensor(0.0, device=attn_lh2rh.device if attn_lh2rh is not None else 'cpu')
        
        # Check if attention has heads dimension
        if attn_lh2rh.dim() == 4:
            # Has heads dimension [batch, heads, len, len]
            # Average over attention heads
            attn_lh2rh = attn_lh2rh.mean(dim=1)  # [batch, len, len]
            attn_rh2lh = attn_rh2lh.mean(dim=1)  # [batch, len, len]
        elif attn_lh2rh.dim() == 3:
            # Already [batch, len, len], no averaging needed
            pass
        else:
            raise ValueError(f"Unexpected attention shape: {attn_lh2rh.shape}")
        
        # Apply temperature and normalize
        attn_lh2rh = F.softmax(attn_lh2rh / self.temperature, dim=-1)
        attn_rh2lh = F.softmax(attn_rh2lh / self.temperature, dim=-1)
        
        # Transpose one attention map for symmetry comparison
        attn_rh2lh_T = attn_rh2lh.transpose(-2, -1)
        
        # KL divergence (bidirectional for stability)
        kl_forward = self.kl_loss(
            F.log_softmax(attn_lh2rh / self.temperature, dim=-1),
            attn_rh2lh_T
        )
        kl_backward = self.kl_loss(
            F.log_softmax(attn_rh2lh_T / self.temperature, dim=-1),
            attn_lh2rh
        )
        
        return (kl_forward + kl_backward) / 2.0
    
    def _compute_alignment_loss(self, lh_feat, rh_feat, body_feat):
        """
        Compute feature alignment loss using cosine similarity.
        
        Encourages features from different body parts to be aligned in the same
        semantic space while maintaining their distinctiveness.
        
        Note: Since lh_feat and rh_feat have the same dimension (42), but body_feat
        has a different dimension (24), we primarily focus on hand-to-hand alignment
        and use L2 distance for hand-to-body alignment.
        
        Args:
            lh_feat: Left hand features [batch, seq_len, dim_lh] or [seq_len, batch, dim_lh]
            rh_feat: Right hand features [batch, seq_len, dim_rh] or [seq_len, batch, dim_rh]
            body_feat: Body features [batch, seq_len, dim_body] or [seq_len, batch, dim_body]
            
        Returns:
            Alignment loss (scalar)
        """
        # Handle both [batch, seq, dim] and [seq, batch, dim] formats
        if lh_feat.dim() == 3 and lh_feat.size(0) < lh_feat.size(1):
            # Likely [seq, batch, dim], transpose to [batch, seq, dim]
            lh_feat = lh_feat.transpose(0, 1)
            rh_feat = rh_feat.transpose(0, 1)
            body_feat = body_feat.transpose(0, 1)
        
        # Global average pooling over sequence dimension
        lh_pooled = lh_feat.mean(dim=1)  # [batch, dim_lh]
        rh_pooled = rh_feat.mean(dim=1)  # [batch, dim_rh]
        body_pooled = body_feat.mean(dim=1)  # [batch, dim_body]
        
        # Normalize features for cosine similarity
        lh_norm = F.normalize(lh_pooled, p=2, dim=-1)
        rh_norm = F.normalize(rh_pooled, p=2, dim=-1)
        
        # 1. Hand-to-hand alignment (cosine similarity - same dimensions)
        sim_lh_rh = (lh_norm * rh_norm).sum(dim=-1).mean()  # Maximize this
        
        # 2. Hand-to-body alignment (L2 distance - different dimensions)
        # We encourage hands and body to have similar activation patterns
        dist_lh_body = F.mse_loss(lh_pooled.std(dim=-1), body_pooled.std(dim=-1))
        dist_rh_body = F.mse_loss(rh_pooled.std(dim=-1), body_pooled.std(dim=-1))
        
        # Alignment loss: negative similarity + distance
        # We want to MAXIMIZE hand-hand similarity and MINIMIZE hand-body variance difference
        alignment_loss = -sim_lh_rh + 0.1 * (dist_lh_body + dist_rh_body)
        
        return alignment_loss
    
    def set_lambda_consistency(self, value):
        """Update consistency loss weight."""
        self.lambda_consistency = value
    
    def set_lambda_alignment(self, value):
        """Update alignment loss weight."""
        self.lambda_alignment = value


class AdaptiveCMCL(CrossModalConsistencyLoss):
    """
    Adaptive CMCL that dynamically adjusts loss weights during training.
    
    The weights start high and gradually decrease as the model learns better
    cross-modal representations.
    """
    
    def __init__(self, num_classes, lambda_consistency=0.1, lambda_alignment=0.05, 
                 temperature=1.0, warmup_epochs=10, decay_rate=0.95):
        super(AdaptiveCMCL, self).__init__(
            num_classes, lambda_consistency, lambda_alignment, temperature
        )
        
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.initial_lambda_consistency = lambda_consistency
        self.initial_lambda_alignment = lambda_alignment
        self.current_epoch = 0
    
    def update_epoch(self, epoch):
        """Update loss weights based on current epoch."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = epoch / self.warmup_epochs
        else:
            # Exponential decay after warmup
            factor = self.decay_rate ** (epoch - self.warmup_epochs)
        
        self.lambda_consistency = self.initial_lambda_consistency * factor
        self.lambda_alignment = self.initial_lambda_alignment * factor
