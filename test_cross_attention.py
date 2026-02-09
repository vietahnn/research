"""
Quick test script to verify Cross-Modal Attention implementation
This script tests the forward pass without requiring actual data
"""

import torch
import sys
sys.path.append('.')

from siformer.cross_modal_attention import CrossModalAttentionFusion, SimplifiedCrossModalAttention
from siformer.model import SiFormer

def test_cross_modal_attention():
    """Test the CrossModalAttentionFusion module"""
    print("=" * 60)
    print("Testing CrossModalAttentionFusion Module")
    print("=" * 60)
    
    # Parameters
    batch_size = 4
    seq_len = 204
    d_lhand = 42
    d_rhand = 42
    d_body = 24
    num_heads = 3  # Changed from 4 to 3 (must divide 42 and 24)
    
    # Create module
    cross_attn = CrossModalAttentionFusion(
        d_lhand=d_lhand,
        d_rhand=d_rhand,
        d_body=d_body,
        num_heads=num_heads,
        dropout=0.1
    )
    
    # Create dummy inputs [Length, Batch, Feature]
    lh_feat = torch.randn(seq_len, batch_size, d_lhand)
    rh_feat = torch.randn(seq_len, batch_size, d_rhand)
    body_feat = torch.randn(seq_len, batch_size, d_body)
    
    print(f"Input shapes:")
    print(f"  Left hand: {lh_feat.shape}")
    print(f"  Right hand: {rh_feat.shape}")
    print(f"  Body: {body_feat.shape}")
    
    # Forward pass
    lh_out, rh_out, body_out = cross_attn(lh_feat, rh_feat, body_feat)
    
    print(f"\nOutput shapes:")
    print(f"  Left hand: {lh_out.shape}")
    print(f"  Right hand: {rh_out.shape}")
    print(f"  Body: {body_out.shape}")
    
    # Check shapes match
    assert lh_out.shape == lh_feat.shape, "Left hand output shape mismatch!"
    assert rh_out.shape == rh_feat.shape, "Right hand output shape mismatch!"
    assert body_out.shape == body_feat.shape, "Body output shape mismatch!"
    
    print("\n✓ Shape validation passed!")
    
    # Check for NaN
    assert not torch.isnan(lh_out).any(), "NaN detected in left hand output!"
    assert not torch.isnan(rh_out).any(), "NaN detected in right hand output!"
    assert not torch.isnan(body_out).any(), "NaN detected in body output!"
    
    print("✓ No NaN values detected!")
    
    # Count parameters
    total_params = sum(p.numel() for p in cross_attn.parameters())
    trainable_params = sum(p.numel() for p in cross_attn.parameters() if p.requires_grad)
    
    print(f"\nModule parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("CrossModalAttentionFusion Test PASSED ✓")
    print("=" * 60 + "\n")


def test_siformer_integration():
    """Test SiFormer model with cross-attention"""
    print("=" * 60)
    print("Testing SiFormer Integration")
    print("=" * 60)
    
    # Parameters
    batch_size = 2
    seq_len = 204
    num_classes = 100
    
    # Create models
    print("\n1. Creating baseline model (no cross-attention)...")
    model_baseline = SiFormer(
        num_classes=num_classes,
        num_hid=108,
        attn_type='prob',
        num_enc_layers=2,  # Reduced for faster testing
        num_dec_layers=1,
        IA_encoder=True,
        IA_decoder=False,
        patience=1,
        use_cross_attention=False
    )
    
    print("2. Creating model with cross-attention...")
    model_cross_attn = SiFormer(
        num_classes=num_classes,
        num_hid=108,
        attn_type='prob',
        num_enc_layers=2,
        num_dec_layers=1,
        IA_encoder=True,
        IA_decoder=False,
        patience=1,
        use_cross_attention=True,
        cross_attn_heads=3  # Changed from 4 to 3
    )
    
    # Create dummy inputs [Batch, Seq, Joints, Coords]
    l_hand = torch.randn(batch_size, seq_len, 21, 2)
    r_hand = torch.randn(batch_size, seq_len, 21, 2)
    body = torch.randn(batch_size, seq_len, 12, 2)
    
    print(f"\nInput shapes:")
    print(f"  Left hand: {l_hand.shape}")
    print(f"  Right hand: {r_hand.shape}")
    print(f"  Body: {body.shape}")
    
    # Forward pass - baseline
    print("\n3. Running baseline model forward pass...")
    model_baseline.eval()
    with torch.no_grad():
        output_baseline = model_baseline(l_hand, r_hand, body, training=False)
    
    print(f"  Baseline output shape: {output_baseline.shape}")
    assert output_baseline.shape == torch.Size([batch_size, num_classes]), "Baseline output shape mismatch!"
    assert not torch.isnan(output_baseline).any(), "NaN in baseline output!"
    
    # Forward pass - with cross-attention
    print("4. Running cross-attention model forward pass...")
    model_cross_attn.eval()
    with torch.no_grad():
        output_cross_attn = model_cross_attn(l_hand, r_hand, body, training=False)
    
    print(f"  Cross-attention output shape: {output_cross_attn.shape}")
    assert output_cross_attn.shape == torch.Size([batch_size, num_classes]), "Cross-attention output shape mismatch!"
    assert not torch.isnan(output_cross_attn).any(), "NaN in cross-attention output!"
    
    # Compare parameter counts
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_cross_attn = sum(p.numel() for p in model_cross_attn.parameters())
    
    print(f"\nParameter comparison:")
    print(f"  Baseline: {params_baseline:,}")
    print(f"  With cross-attention: {params_cross_attn:,}")
    print(f"  Increase: {params_cross_attn - params_baseline:,} ({(params_cross_attn/params_baseline - 1)*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("SiFormer Integration Test PASSED ✓")
    print("=" * 60 + "\n")


def test_backward_pass():
    """Test backward pass and gradient flow"""
    print("=" * 60)
    print("Testing Backward Pass and Gradients")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 204
    num_classes = 10  # Smaller for faster testing
    
    # Create model
    model = SiFormer(
        num_classes=num_classes,
        num_hid=108,
        num_enc_layers=2,
        num_dec_layers=1,
        use_cross_attention=True,
        cross_attn_heads=3  # Changed from 2 to 3
    )
    
    # Create dummy inputs
    l_hand = torch.randn(batch_size, seq_len, 21, 2)
    r_hand = torch.randn(batch_size, seq_len, 21, 2)
    body = torch.randn(batch_size, seq_len, 12, 2)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    model.train()
    outputs = model(l_hand, r_hand, body, training=True)
    
    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}!"
    
    assert has_gradients, "No gradients computed!"
    
    print("✓ Backward pass successful!")
    print("✓ All gradients are valid (no NaN)!")
    
    print("\n" + "=" * 60)
    print("Backward Pass Test PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Cross-Modal Attention Implementation Verification")
    print("=" * 60 + "\n")
    
    try:
        # Run all tests
        test_cross_modal_attention()
        test_siformer_integration()
        test_backward_pass()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        print("\nImplementation is ready for training!")
        print("You can now run: python train.py --use_cross_attention True")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
