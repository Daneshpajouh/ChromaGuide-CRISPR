import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_platform.knowledge_retriever import KnowledgeRetriever

async def test_web_search():
    print("🚀 Initializing KnowledgeRetriever with Web Search...")
    kr = KnowledgeRetriever()

    topic = "Recent breakthroughs in Mamba-2 for sequence modeling"
    print(f"📡 Dispatching unified search for: {topic}")

    results = kr.fetch_academic_context(topic, limit=5)
    print("\n--- Consolidated Results ---")
    print(results)

    if "web" in results.lower() or "global web" in results.lower():
        print("\n✅ Web Search results detected in consolidation!")
    else:
        print("\n❌ Web Search results not found in consolidation.")

if __name__ == "__main__":
    asyncio.run(test_web_search())
