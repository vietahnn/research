"""
Quick test script to verify Spatial-Temporal Positional Encoding implementation.

This script tests:
1. SpatialTemporalPE module works correctly
2. SiFormer model integrates spatial PE properly
3. Forward pass completes without errors
"""

import torch
from siformer.positional_encoding import SpatialTemporalPE
from siformer.model import SiFormer


def test_spatial_temporal_pe():
    """Test the SpatialTemporalPE module directly"""
    print("=" * 60)
    print("Test 1: SpatialTemporalPE Module")
    print("=" * 60)
    
    # Test for hand (21 joints)
    pe_hand = SpatialTemporalPE(
        num_joints=21, 
        d_coords=2, 
        seq_len=204, 
        encoding_type='learnable'
    )
    
    # Create dummy input: (seq_len=100, batch_size=4, d_model=42)
    x = torch.randn(100, 4, 42)
    output = pe_hand(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Temporal PE shape: {pe_hand.temporal_pe.shape}")
    print(f"✓ Spatial PE shape: {pe_hand.spatial_pe.shape}")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    print("✓ SpatialTemporalPE test passed!\n")
    
    # Test sinusoidal variant
    print("Testing sinusoidal encoding (both spatial & temporal)...")
    pe_hand_sin = SpatialTemporalPE(
        num_joints=21, 
        d_coords=2, 
        seq_len=204, 
        encoding_type='sinusoidal'
    )
    output_sin = pe_hand_sin(x)
    assert output_sin.shape == x.shape, "Sinusoidal output shape mismatch!"
    print("✓ Sinusoidal encoding (both spatial & temporal) test passed!\n")


def test_siformer_with_spatial_pe():
    """Test SiFormer model with spatial-temporal PE"""
    print("=" * 60)
    print("Test 2: SiFormer with Spatial-Temporal PE")
    print("=" * 60)
    
    # Create model with learnable PE (default)
    model = SiFormer(
        num_classes=100,
        num_hid=108,
        num_enc_layers=2,
        num_dec_layers=1,
        pe_type='learnable'
    )
    
    print(f"✓ Model created with learnable PE (both spatial & temporal)")
    print(f"✓ L-hand embedding type: {type(model.l_hand_embedding)}")
    print(f"✓ R-hand embedding type: {type(model.r_hand_embedding)}")
    print(f"✓ Body embedding type: {type(model.body_embedding)}")
    
    # Create dummy data
    batch_size = 4
    seq_len = 100
    l_hand = torch.randn(batch_size, seq_len, 21, 2)
    r_hand = torch.randn(batch_size, seq_len, 21, 2)
    body = torch.randn(batch_size, seq_len, 12, 2)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(l_hand, r_hand, body, training=False)
    
    print(f"✓ Forward pass completed")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (batch_size, 100), "Output shape mismatch!"
    print("✓ SiFormer with spatial-temporal PE test passed!\n")


def test_siformer_sinusoidal():
    """Test SiFormer model with sinusoidal PE"""
    print("=" * 60)
    print("Test 3: SiFormer with Sinusoidal PE (Backward Compatibility)")
    print("=" * 60)
    
    # Create model with sinusoidal PE
    model = SiFormer(
        num_classes=100,
        num_hid=108,
        num_enc_layers=2,
        num_dec_layers=1,
        pe_type='sinusoidal'
    )
    
    print(f"✓ Model created with sinusoidal PE (both spatial & temporal)")
    print(f"✓ L-hand embedding type: {type(model.l_hand_embedding)}")
    
    # Create dummy data
    batch_size = 4
    seq_len = 100
    l_hand = torch.randn(batch_size, seq_len, 21, 2)
    r_hand = torch.randn(batch_size, seq_len, 21, 2)
    body = torch.randn(batch_size, seq_len, 12, 2)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(l_hand, r_hand, body, training=False)
    
    print(f"✓ Forward pass completed")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (batch_size, 100), "Output shape mismatch!"
    print("✓ Sinusoidal PE test passed!\n")


def test_parameter_count():
    """Compare parameter counts between learnable and sinusoidal PE"""
    print("=" * 60)
    print("Test 4: Parameter Count Comparison")
    print("=" * 60)
    
    # Model with learnable PE
    model_learnable = SiFormer(
        num_classes=100,
        num_hid=108,
        num_enc_layers=2,
        num_dec_layers=1,
        pe_type='learnable'
    )
    
    # Model with sinusoidal PE
    model_sinusoidal = SiFormer(
        num_classes=100,
        num_hid=108,
        num_enc_layers=2,
        num_dec_layers=1,
        pe_type='sinusoidal'
    )
    
    params_learnable = sum(p.numel() for p in model_learnable.parameters())
    params_sinusoidal = sum(p.numel() for p in model_sinusoidal.parameters())
    
    print(f"Parameters (learnable PE): {params_learnable:,}")
    print(f"Parameters (sinusoidal PE): {params_sinusoidal:,}")
    print(f"Difference: {abs(params_learnable - params_sinusoidal):,}")
    print(f"Percentage difference: {abs(params_learnable - params_sinusoidal) / params_sinusoidal * 100:.2f}%")
    print("✓ Parameter count comparison completed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPATIAL-TEMPORAL POSITIONAL ENCODING TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_spatial_temporal_pe()
        test_siformer_with_spatial_pe()
        test_siformer_sinusoidal()
        test_parameter_count()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nYou can now train with different PE types:")
        print("  python train.py --pe_type learnable   (default)")
        print("  python train.py --pe_type sinusoidal")
        print("\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED! ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
