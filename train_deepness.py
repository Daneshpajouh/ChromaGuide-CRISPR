"""Compatibility wrapper for `src/train_deepmens.py`.

Some scripts call `train_deepness.py` (typo). This wrapper forwards CLI args
to the real `src/train_deepmens.py` entrypoint when available, or attempts to
import and call a `train_deepmens_model` entrypoint if provided.
"""
import argparse
import subprocess
import sys


def delegate_cli():
    parser = argparse.ArgumentParser(description="Compatibility wrapper for train_deepmens")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_mini", action="store_true")
    parser.add_argument("--output_dir", type=str, default="models/deepmens")
    args = parser.parse_args()

    # Try to import and call a function from src.train_deepmens if available
    try:
        sys.path.append("src")
        from src.train_deepmens import train_deepmens_model
        try:
            train_deepmens_model(args)
            return
        except Exception:
            # Fall through to subprocess fallback
            pass
    except Exception:
        pass

    # Fallback: call the script directly if present
    try:
        subprocess.run([sys.executable, "src/train_deepmens.py",
                        "--seed", str(args.seed),
                        "--epochs", str(args.epochs),
                        "--batch_size", str(args.batch_size),
                        "--output_dir", args.output_dir], check=False)
    except Exception as e:
        print("Failed to delegate to src/train_deepmens.py:", e)


if __name__ == "__main__":
    delegate_cli()
