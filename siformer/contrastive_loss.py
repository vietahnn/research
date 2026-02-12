"""
Cross-Modal Contrastive Learning for Sign Language Recognition

This module implements contrastive loss to enforce semantic alignment between
different body parts (left hand, right hand, body) within the same sign class.

Uses InfoNCE (Normalized Temperature-scaled Cross Entropy) loss:
- Positive pairs: Same sign label → features should be aligned
- Negative pairs: Different sign labels → features should be distinct

Author: Research Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    """
    Contrastive loss to align body part features within same sign class.
    
    Enforces that for the same sign:
    - Left hand ↔ Right hand features are consistent
    - Left hand ↔ Body features are consistent  
    - Right hand ↔ Body features are consistent
    
    For different signs, features should be distinctive.
    
    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
        projection_dim: Dimension of projection space (default: 128)
        d_lhand: Dimension of left hand features (default: 42)
        d_rhand: Dimension of right hand features (default: 42)
        d_body: Dimension of body features (default: 24)
    
    Example:
        >>> criterion = CrossModalContrastiveLoss()
        >>> lh_feat = torch.randn(24, 204, 42)  # [Batch, Seq, Dim]
        >>> rh_feat = torch.randn(24, 204, 42)
        >>> body_feat = torch.randn(24, 204, 24)
        >>> labels = torch.randint(0, 100, (24,))
        >>> loss = criterion(lh_feat, rh_feat, body_feat, labels)
    """
    
    def __init__(self, temperature=0.07, projection_dim=128, 
                 d_lhand=42, d_rhand=42, d_body=24):
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim
        
        # Projection heads to map features to unified embedding space
        self.lh_proj = self._create_projection_head(d_lhand, projection_dim)
        self.rh_proj = self._create_projection_head(d_rhand, projection_dim)
        self.body_proj = self._create_projection_head(d_body, projection_dim)
        
        print(f"CrossModalContrastiveLoss initialized:")
        print(f"  Temperature: {temperature}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Left hand: {d_lhand} → {projection_dim}")
        print(f"  Right hand: {d_rhand} → {projection_dim}")
        print(f"  Body: {d_body} → {projection_dim}")
    
    def _create_projection_head(self, input_dim, output_dim):
        """
        Create MLP projection head.
        
        2-layer MLP with ReLU activation and LayerNorm.
        Projects features to unified embedding space for comparison.
        """
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def temporal_pool(self, features):
        """
        Average pooling over temporal dimension.
        
        Args:
            features: [Batch, SeqLen, Dim] or [SeqLen, Batch, Dim]
        
        Returns:
            pooled: [Batch, Dim]
        """
        # Handle both formats
        if features.dim() == 3:
            if features.size(0) > features.size(1):  # Likely [SeqLen, Batch, Dim]
                features = features.transpose(0, 1)  # -> [Batch, SeqLen, Dim]
        
        # Average across time
        pooled = features.mean(dim=1)  # [Batch, Dim]
        return pooled
    
    def info_nce_loss(self, anchor, positive, labels):
        """
        InfoNCE (Normalized Temperature-scaled Cross Entropy) loss.
        
        For each anchor, maximize similarity to positives (same class)
        and minimize similarity to negatives (different class).
        
        Args:
            anchor: [Batch, Dim] - query embeddings
            positive: [Batch, Dim] - key embeddings (different modality)
            labels: [Batch] - class labels
        
        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = anchor.size(0)
        
        # L2 normalize embeddings (cosine similarity)
        anchor = F.normalize(anchor, dim=-1, p=2)
        positive = F.normalize(positive, dim=-1, p=2)
        
        # Compute similarity matrix: anchor[i] · positive[j]
        # Shape: [Batch, Batch]
        similarity_matrix = torch.matmul(anchor, positive.T) / self.temperature
        
        # Create label mask: same_class[i,j] = 1 if labels[i] == labels[j], else 0
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(similarity_matrix.device)
        
        # Remove diagonal (self-similarity)
        mask.fill_diagonal_(0)
        
        # For numerical stability, use max subtraction
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute exp
        exp_logits = torch.exp(logits)
        
        # Positive pairs
        # For samples with no positive pairs (unique class in batch), we skip them
        positives_per_sample = mask.sum(dim=1)
        
        # Compute log probability
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)
        
        # Only compute loss for samples that have positive pairs
        # Avoid division by zero
        valid_samples = positives_per_sample > 0
        if valid_samples.sum() > 0:
            mean_log_prob_pos = mean_log_prob_pos[valid_samples] / positives_per_sample[valid_samples]
            loss = -mean_log_prob_pos.mean()
        else:
            # If no valid positive pairs in batch, return zero loss
            loss = torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
        return loss
    
    def forward(self, lh_feat, rh_feat, body_feat, labels):
        """
        Compute cross-modal contrastive loss.
        
        Args:
            lh_feat: Left hand features [Batch, SeqLen, d_lhand] or [SeqLen, Batch, d_lhand]
            rh_feat: Right hand features [Batch, SeqLen, d_rhand] or [SeqLen, Batch, d_rhand]
            body_feat: Body features [Batch, SeqLen, d_body] or [SeqLen, Batch, d_body]
            labels: Class labels [Batch]
        
        Returns:
            loss: Scalar contrastive loss
        """
        # Temporal pooling to get sequence-level representations
        lh_pooled = self.temporal_pool(lh_feat)  # [Batch, d_lhand]
        rh_pooled = self.temporal_pool(rh_feat)  # [Batch, d_rhand]
        body_pooled = self.temporal_pool(body_feat)  # [Batch, d_body]
        
        # Project to unified embedding space
        lh_emb = self.lh_proj(lh_pooled)  # [Batch, projection_dim]
        rh_emb = self.rh_proj(rh_pooled)  # [Batch, projection_dim]
        body_emb = self.body_proj(body_pooled)  # [Batch, projection_dim]
        
        # Compute three pairwise contrastive losses
        # 1. Left hand ↔ Right hand alignment
        loss_lh_rh = self.info_nce_loss(lh_emb, rh_emb, labels)
        
        # 2. Left hand ↔ Body alignment
        loss_lh_body = self.info_nce_loss(lh_emb, body_emb, labels)
        
        # 3. Right hand ↔ Body alignment
        loss_rh_body = self.info_nce_loss(rh_emb, body_emb, labels)
        
        # Average the three losses
        total_loss = (loss_lh_rh + loss_lh_body + loss_rh_body) / 3.0
        
        return total_loss


class SimplifiedContrastiveLoss(nn.Module):
    """
    Simplified version for ablation studies.
    Only aligns hands with body (uni-directional).
    """
    
    def __init__(self, temperature=0.07, projection_dim=128,
                 d_lhand=42, d_rhand=42, d_body=24):
        super(SimplifiedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
        # Only body projection (hands align to body)
        self.body_proj = nn.Sequential(
            nn.Linear(d_body, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.lh_proj = nn.Sequential(
            nn.Linear(d_lhand, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.rh_proj = nn.Sequential(
            nn.Linear(d_rhand, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, lh_feat, rh_feat, body_feat, labels):
        """Simplified contrastive loss (hands → body only)."""
        # Temporal pooling
        lh_pooled = lh_feat.mean(dim=1) if lh_feat.dim() == 3 else lh_feat.transpose(0, 1).mean(dim=1)
        rh_pooled = rh_feat.mean(dim=1) if rh_feat.dim() == 3 else rh_feat.transpose(0, 1).mean(dim=1)
        body_pooled = body_feat.mean(dim=1) if body_feat.dim() == 3 else body_feat.transpose(0, 1).mean(dim=1)
        
        # Project
        lh_emb = F.normalize(self.lh_proj(lh_pooled), dim=-1)
        rh_emb = F.normalize(self.rh_proj(rh_pooled), dim=-1)
        body_emb = F.normalize(self.body_proj(body_pooled), dim=-1)
        
        # Simple cosine similarity loss
        loss_lh = 1 - F.cosine_similarity(lh_emb, body_emb).mean()
        loss_rh = 1 - F.cosine_similarity(rh_emb, body_emb).mean()
        
        return (loss_lh + loss_rh) / 2.0
