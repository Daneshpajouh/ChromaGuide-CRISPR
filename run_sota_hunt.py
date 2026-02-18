import sys
import os

# Align path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from research_platform.master_stack import ResearchMaster

def run_hunt():
    print("🚀 Initiating SOTA 'Small & Smart' Model Hunt...")
    master = ResearchMaster()

    # The Directive Topic for the Agents
    topic = "State-of-the-Art Multimodal Small Language Models under 8B parameters 2025 (Vision Qwen2.5, Genomics Mamba, Audio Whisper)"

    # Execute the research cycle
    report_path = master.investigate(topic)

    print(f"\n✅ Hunt Complete. Report generated at: {report_path}")

if __name__ == "__main__":
    run_hunt()
