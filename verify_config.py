from src.model.mamba2_block import Mamba2Config

try:
    cfg = Mamba2Config(
        d_model=256,
        ssm_cfg={"dt_min": 0.001}
    )
    print("SUCCESS: Mamba2Config accepts ssm_cfg")
    print(f"ssm_cfg: {cfg.ssm_cfg}")
except Exception as e:
    print(f"FAILURE: {e}")
    exit(1)
