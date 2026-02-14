"""
Test script to verify Three Pairs Uni-directional Cross-Modal Attention
"""

import torch
import sys
sys.path.append('.')

from siformer.cross_modal_attention import UniDirectionalCrossModalAttention

def test_three_pairs_attention():
    """Test the three pairs uni-directional attention"""
    print("=" * 70)
    print("Testing Three Pairs Uni-directional Cross-Modal Attention")
    print("Three independent attentions:")
    print("  1. Left Hand → Body")
    print("  2. Right Hand → Body")
    print("  3. Right Hand → Left Hand")
    print("=" * 70)
    
    # Parameters
    batch_size = 4
    seq_len = 204
    d_lhand = 42
    d_rhand = 42
    d_body = 24
    num_heads = 3  # Must divide 42 and 24
    
    # Create module with 'three_pairs' direction
    three_pairs_attn = UniDirectionalCrossModalAttention(
        d_lhand=d_lhand,
        d_rhand=d_rhand,
        d_body=d_body,
        num_heads=num_heads,
        dropout=0.1,
        direction='three_pairs'  # New three_pairs direction
    )
    
    # Create dummy inputs [Length, Batch, Feature]
    lh_feat = torch.randn(seq_len, batch_size, d_lhand)
    rh_feat = torch.randn(seq_len, batch_size, d_rhand)
    body_feat = torch.randn(seq_len, batch_size, d_body)
    
    print(f"\n{'='*70}")
    print("Input shapes:")
    print(f"{'='*70}")
    print(f"  Left hand:  {lh_feat.shape}")
    print(f"  Right hand: {rh_feat.shape}")
    print(f"  Body:       {body_feat.shape}")
    
    # Forward pass
    print(f"\n{'='*70}")
    print("Forward pass...")
    print(f"{'='*70}")
    lh_out, rh_out, body_out = three_pairs_attn(lh_feat, rh_feat, body_feat)
    
    print(f"\nOutput shapes:")
    print(f"  Left hand:  {lh_out.shape}")
    print(f"  Right hand: {rh_out.shape}")
    print(f"  Body:       {body_out.shape}")
    
    # Verify shapes
    assert lh_out.shape == lh_feat.shape, "Left hand output shape mismatch!"
    assert rh_out.shape == rh_feat.shape, "Right hand output shape mismatch!"
    assert body_out.shape == body_feat.shape, "Body output shape mismatch!"
    print("\n✓ Shape validation passed!")
    
    # Check for NaN
    assert not torch.isnan(lh_out).any(), "NaN detected in left hand output!"
    assert not torch.isnan(rh_out).any(), "NaN detected in right hand output!"
    assert not torch.isnan(body_out).any(), "NaN detected in body output!"
    print("✓ No NaN values detected!")
    
    # Verify outputs are different from inputs (attention had an effect)
    lh_changed = not torch.allclose(lh_out, lh_feat)
    rh_changed = not torch.allclose(rh_out, rh_feat)
    body_changed = not torch.allclose(body_out, body_feat)
    
    print(f"\nAttention effects:")
    print(f"  Left hand modified:  {lh_changed}")
    print(f"  Right hand modified: {rh_changed}")
    print(f"  Body modified:       {body_changed}")
    
    assert lh_changed, "Left hand should be modified by attention!"
    assert rh_changed, "Right hand should be modified by attention!"
    assert body_changed == False, "Body should NOT be modified (no attention to body)!"
    print("✓ Left hand and right hand were enhanced, body unchanged!")
    
    # Count parameters
    total_params = sum(p.numel() for p in three_pairs_attn.parameters())
    trainable_params = sum(p.numel() for p in three_pairs_attn.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print("Module parameters:")
    print(f"{'='*70}")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Show attention modules
    print(f"\n{'='*70}")
    print("Three Pairs Attention Architecture:")
    print(f"{'='*70}")
    print("  1. LH attends to Body     (lh_attn + body_to_lh_proj)")
    print("  2. RH attends to Body     (rh_body_attn + body_to_rh_proj)")
    print("  3. RH attends to LH       (rh_lh_attn + lh_to_rh_proj)")
    print("\nResult:")
    print("  - Left hand:  Enhanced by body context")
    print("  - Right hand: Enhanced by body + left hand")
    print("  - Body:       Unchanged (no incoming attention)")
    
    print("\n" + "=" * 70)
    print("Three Pairs Uni-directional Attention Test PASSED ✓")
    print("=" * 70 + "\n")


def test_backward_three_pairs():
    """Test backward pass with three pairs attention"""
    print("=" * 70)
    print("Testing Backward Pass with Three Pairs Attention")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 204
    d_lhand = 42
    d_rhand = 42
    d_body = 24
    
    # Create module
    three_pairs_attn = UniDirectionalCrossModalAttention(
        d_lhand=d_lhand,
        d_rhand=d_rhand,
        d_body=d_body,
        num_heads=3,
        dropout=0.1,
        direction='three_pairs'
    )
    
    # Create inputs with gradient tracking
    lh_feat = torch.randn(seq_len, batch_size, d_lhand, requires_grad=True)
    rh_feat = torch.randn(seq_len, batch_size, d_rhand, requires_grad=True)
    body_feat = torch.randn(seq_len, batch_size, d_body, requires_grad=True)
    
    # Forward pass
    lh_out, rh_out, body_out = three_pairs_attn(lh_feat, rh_feat, body_feat)
    
    # Compute dummy loss
    loss = lh_out.sum() + rh_out.sum() + body_out.sum()
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    assert lh_feat.grad is not None, "No gradient for left hand!"
    assert rh_feat.grad is not None, "No gradient for right hand!"
    assert body_feat.grad is not None, "No gradient for body!"
    
    assert not torch.isnan(lh_feat.grad).any(), "NaN gradient in left hand!"
    assert not torch.isnan(rh_feat.grad).any(), "NaN gradient in right hand!"
    assert not torch.isnan(body_feat.grad).any(), "NaN gradient in body!"
    
    print("✓ Backward pass successful!")
    print("✓ All gradients are valid (no NaN)!")
    
    # Check parameter gradients
    param_with_grad = 0
    for name, param in three_pairs_attn.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}!"
            param_with_grad += 1
    
    print(f"✓ {param_with_grad} parameters have valid gradients!")
    
    print("\n" + "=" * 70)
    print("Backward Pass Test PASSED ✓")
    print("=" * 70 + "\n")


def compare_attention_strategies():
    """Compare parameter counts for different strategies"""
    print("=" * 70)
    print("Comparing Different Attention Strategies")
    print("=" * 70)
    
    d_lhand = 42
    d_rhand = 42
    d_body = 24
    num_heads = 3
    
    strategies = {
        'hands_to_body': 'Hands → Body',
        'body_to_hands': 'Body → Hands',
        'hands_bidirectional': 'LH ↔ RH',
        'chain': 'LH→RH→Body→LH',
        'three_pairs': 'LH→Body, RH→Body, RH→LH'
    }
    
    print(f"\n{'Strategy':<25} {'Description':<30} {'Parameters':>15}")
    print("-" * 70)
    
    for direction, desc in strategies.items():
        model = UniDirectionalCrossModalAttention(
            d_lhand=d_lhand,
            d_rhand=d_rhand,
            d_body=d_body,
            num_heads=num_heads,
            dropout=0.1,
            direction=direction
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"{direction:<25} {desc:<30} {params:>15,}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("THREE PAIRS UNI-DIRECTIONAL CROSS-MODAL ATTENTION VERIFICATION")
    print("=" * 70 + "\n")
    
    try:
        # Run tests
        test_three_pairs_attention()
        test_backward_three_pairs()
        compare_attention_strategies()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓✓✓")
        print("=" * 70)
        print("\nThree pairs attention is ready for use!")
        print("\nTo use in training:")
        print("  python train.py \\")
        print("    --use_cross_attention True \\")
        print("    --cross_attn_direction three_pairs \\")
        print("    --cross_attn_heads 3")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ✗")
        print("=" * 70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
