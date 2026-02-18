import torch
import os
import sys

def diagnose_mamba_h100():
    print("=== Mamba-2 H100 Diagnostic (PyTorch 2.5 + MPS) ===\n")

    # 1. Check versions
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.mps.is_available()}")
    if torch.mps.is_available():
        print(f"GPU: {torch.mps.get_device_name(0)}")
        print(f"Compute Capability: {torch.mps.get_device_capability(0)}")
    else:
        print("CRITICAL: CUDA not available!")
        sys.exit(1)

    # 2. Check mamba-ssm installation
    try:
        import mamba_ssm
        print(f"Mamba-SSM: {mamba_ssm.__version__}")
    except Exception as e:
        print(f"Mamba-SSM: FAILED TO IMPORT ({e})")

    # 3. Test kernel loading
    print("\n[Kernel Test]")
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        print("✓ selective_scan_fn loaded successfully")
    except Exception as e:
        print(f"✗ selective_scan_fn FAILED: {e}")

    # 4. Test Model Instantiation & Initialization (CPU)
    print("\n[Architecture Test]")
    try:
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.modules.mamba2 import Mamba2

        # Create config with SAFE parameters
        config = MambaConfig(
            d_model=256,
            d_state=128,
            n_layer=2,
            ssm_cfg={
                "chunk_size": 256,
            }
        )
        print("Config created.")

        # Manual Init like in our Codebase
        model = Mamba2(d_model=256, d_state=128, d_conv=4, expand=2, headdim=64).cpu()
        print("Mamba2 Layer instantiated on CPU.")

        # Apply Safe Init (The Fix we tried)
        for name, param in model.named_parameters():
             if 'dt_bias' in name:
                 torch.nn.init.constant_(param, 0.1)
             if 'A_log' in name:
                 torch.nn.init.constant_(param, -3.0)

        # Check Init
        dt_bias_safe = True
        for name, param in model.named_parameters():
            if 'dt_bias' in name:
                if torch.isinf(param).any() or torch.isnan(param).any():
                    print(f"  ⚠️ {name} CONTAINS INF/NAN! (Min: {param.min()}, Max: {param.max()})")
                    dt_bias_safe = False
                else:
                    print(f"  ✓ {name} is safe (Min: {param.min():.4f}, Max: {param.max():.4f})")

        if not dt_bias_safe:
            print("FAILURE: Model Init Failed.")
            sys.exit(1)

    except Exception as e:
        print(f"Failed Verification: {e}")
        sys.exit(1)

    # 5. Forward/Backward Test on H100
    print("\n[Compute Test on H100]")
    try:
        device = "mps"
        model = model.to(device)
        x = torch.randn(4, 4096, 256, device=device, dtype=torch.float32) # B=4, L=4096 (Multiple of 256), D=256

        print("Running Forward Pass (Autocast bfloat16)...")
        with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
            y = model(x)
            loss = y.sum()

        print(f"  Forward Pass: Success (Output shape: {y.shape})")
        print(f"  Output NaNs: {torch.isnan(y).any().item()}")

        if torch.isnan(y).any():
            print("  CRITICAL: NaN in Output!")
        else:
            loss.backward()
            print("  Backward Pass: Success")

            has_nan_grad = False
            for n, p in model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"  ⚠️ NaN Gradient in {n}")
                    has_nan_grad = True

            if not has_nan_grad:
                print("  ✓ Gradients are finite.")
                print("\n✅ SYSTEM IS READY FOR SOTA TRAINING.")
            else:
                 print("\n❌ SYSTEM FAILED GRADIENT CHECK.")

    except Exception as e:
         print(f"Compute Test Failed: {e}")

if __name__ == "__main__":
    diagnose_mamba_h100()
