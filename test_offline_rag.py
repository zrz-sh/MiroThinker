#!/usr/bin/env python3
"""
Test script for offline RAG MCP server.

This script tests the offline_search and offline_access tools
provided by the offline RAG MCP server.
"""

import asyncio
import json
import os
import sys

# Add the miroflow_tools source to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "libs/miroflow-tools/src",
    ),
)

from miroflow_tools.mcp_servers.offline_rag_mcp_server import (
    offline_search,
    offline_access,
)


async def test_offline_search():
    """Test the offline_search tool."""
    print("=" * 60)
    print("Testing offline_search tool")
    print("=" * 60)

    # Test query
    query = "What is the capital of France?"
    print(f"\nQuery: {query}")

    try:
        result = await offline_search(q=query, num=3)
        result_data = json.loads(result)

        print(f"\nSuccess: {result_data.get('success', False)}")
        print(f"Number of results: {result_data.get('num_results', 0)}")

        if result_data.get("success") and result_data.get("results"):
            for item in result_data["results"]:
                print(f"\n[Result {item['rank']}]")
                print(f"URL: {item['url']}")
                print(f"Snippet: {item['snippet'][:200]}...")
        else:
            print(f"\nError or no results: {result_data.get('error', 'Unknown')}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()


async def test_offline_access():
    """Test the offline_access tool."""
    print("\n" + "=" * 60)
    print("Testing offline_access tool")
    print("=" * 60)

    # Test URL
    url = "https://en.wikipedia.org/w/index.php?title=List%20of%20capitals%20of%20France"
    print(f"\nURL: {url}")

    try:
        result = await offline_access(url=url)

        if result.startswith("[ERROR]"):
            print(f"\nError: {result}")
        else:
            print(f"\nSuccess! Retrieved {len(result)} characters")
            print(f"Preview:\n{result[:500]}...")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests."""
    # Set environment variable for testing
    os.environ["RAG_SERVER_ADDR"] = "127.0.0.1:8000"

    print("\n")
    print("=" * 60)
    print("Offline RAG MCP Server Test")
    print("=" * 60)
    print(f"RAG Server: {os.environ.get('RAG_SERVER_ADDR')}")
    print("=" * 60)

    await test_offline_search()
    await test_offline_access()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
