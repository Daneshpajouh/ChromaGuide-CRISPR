import asyncio
import sys
import os
import time

# Align path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_platform.master_stack import ResearchMaster

async def test_async_investigation():
    print("⚡ [TEST] Initializing Async Core...")
    master = ResearchMaster()

    # Mock log callback
    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    print("\n⚡ [TEST] Launching 'Async M5' Investigation...")
    # Using 'ultra_light_fast' to speed up test
    report_path = await master.investigate(
        topic="Mamba-5 Architecture for Genomic Sequencing",
        tier="ultra_light_fast",
        log_callback=log
    )

    print(f"\n✅ [TEST] Async Investigation Complete. Report: {report_path}")
    assert os.path.exists(report_path)

    # Verify report content
    with open(report_path, "r") as f:
        content = f.read()
        if "Mamba-5" in content and "Genomic" in content:
            print("✅ [TEST] Report Content Verified.")
        else:
            print("⚠️ [TEST] Report content missing keywords.")

if __name__ == "__main__":
    asyncio.run(test_async_investigation())
