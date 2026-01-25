# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Offline RAG MCP Server for local search and access.

This module provides tools for:
- Searching documents using a local RAG server
- Accessing webpage content via local RAG server
- Extracting information from webpages using LLM
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List

import aiohttp
import httpx
from mcp.server.fastmcp import FastMCP
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure logging
logger = logging.getLogger("miroflow")

# Local RAG server configuration
RAG_SERVER_ADDR = os.getenv("RAG_SERVER_ADDR", "127.0.0.1:8000")

# Summary LLM configuration (for extracting info from accessed pages)
SUMMARY_LLM_BASE_URL = os.getenv("SUMMARY_LLM_BASE_URL")
SUMMARY_LLM_MODEL_NAME = os.getenv("SUMMARY_LLM_MODEL_NAME")
SUMMARY_LLM_API_KEY = os.getenv("SUMMARY_LLM_API_KEY")

# Initialize FastMCP server
mcp = FastMCP("offline_rag")


class AsyncSearchClient:
    """Local/offline search client that connects to a local RAG server."""

    # Class-level shared session for connection pooling
    _shared_session = None
    _session_lock = threading.Lock()

    @classmethod
    async def get_session(cls):
        """Get or create shared aiohttp session with connection pooling."""
        if cls._shared_session is None or cls._shared_session.closed:
            with cls._session_lock:
                if cls._shared_session is None or cls._shared_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=2000,  # Max total connections
                        limit_per_host=1000,  # Max connections per host
                        ttl_dns_cache=1000,  # DNS cache TTL
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=300, sock_connect=100),
                        trust_env=False,
                    )
        return cls._shared_session

    def __init__(self, server_addr: str):
        self.server_addr = server_addr
        logger.info(f"AsyncSearchClient: Using local server at {self.server_addr}")

    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        """Query local search server."""
        cnt = 0
        last_exception = None
        session = await self.get_session()

        while cnt < 10:
            try:
                async with session.post(
                    f"http://{self.server_addr}/retrieve",
                    json=req_meta,
                ) as response:
                    response.raise_for_status()
                    res = await response.json()
                    return [
                        dict(
                            documents=[r["contents"] for r in result],
                            urls=[r["url"] for r in result],
                            server_type="offline-rag",
                        )
                        for result in res["result"]
                    ]
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"AsyncSearchClient: Search query error {e}. Retry {cnt} times."
                )
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError(
            "Fail to post search query to RAG server"
        ) from last_exception

    async def access_async(self, urls: List[str]) -> List[Dict]:
        """Access URLs via local server."""
        cnt = 0
        last_exception = None
        session = await self.get_session()

        while cnt < 10:
            try:
                async with session.post(
                    f"http://{self.server_addr}/access",
                    json={"urls": urls},
                ) as response:
                    response.raise_for_status()
                    res = await response.json()
                    return [
                        dict(
                            page=result["contents"] if result is not None else "",
                            type="access",
                            server_type="offline-rag",
                        )
                        for result in res["result"]
                    ]
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"AsyncSearchClient: Access request error {e}. Retry {cnt} times."
                )
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError(
            "Fail to post access request to RAG server"
        ) from last_exception


# Global search client instance
search_client = AsyncSearchClient(RAG_SERVER_ADDR)


@mcp.tool()
async def offline_search(
    q: str,
    num: int = 3,
) -> str:
    """
    Tool to perform offline RAG searches using a local search server.

    This tool queries a local RAG (Retrieval-Augmented Generation) server
    to search through pre-indexed documents (e.g., Wikipedia).

    Args:
        q: Search query string (Required)
        num: Number of search results to return (default: 3)

    Returns:
        JSON string containing search results with URLs and document snippets.
    """
    # Check for RAG server address
    if not RAG_SERVER_ADDR:
        return json.dumps(
            {
                "success": False,
                "error": "RAG_SERVER_ADDR environment variable not set",
                "results": [],
            },
            ensure_ascii=False,
        )

    # Validate required parameter
    if not q or not q.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Search query 'q' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    try:
        # Prepare request metadata
        req_meta = {
            "queries": [q.strip()],
            "topk": num,
            "return_scores": False,
        }

        # Query the local RAG server
        results = await search_client.query_async(req_meta)

        if not results or not results[0].get("documents"):
            return json.dumps(
                {
                    "success": True,
                    "message": "No search results found",
                    "results": [],
                },
                ensure_ascii=False,
            )

        # Format results
        documents = results[0]["documents"]
        urls = results[0]["urls"]

        formatted_results = []
        for i, (doc, url) in enumerate(zip(documents, urls), 1):
            formatted_results.append(
                {
                    "rank": i,
                    "url": url,
                    "snippet": doc[:500] + "..." if len(doc) > 500 else doc,
                }
            )

        response_data = {
            "success": True,
            "query": q,
            "num_results": len(formatted_results),
            "results": formatted_results,
        }

        return json.dumps(response_data, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Offline search error: {str(e)}")
        return json.dumps(
            {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "results": [],
            },
            ensure_ascii=False,
        )


@mcp.tool()
async def offline_access(url: str) -> str:
    """
    Tool to access webpage content via local RAG server.

    This tool retrieves the full content of a specific URL from the local
    RAG server's indexed database.

    Args:
        url: The URL to access (must be in the RAG server's index)

    Returns:
        The webpage content as a string.
    """
    # Check for RAG server address
    if not RAG_SERVER_ADDR:
        return "[ERROR]: RAG_SERVER_ADDR environment variable not set. offline_access tool is not available."

    # Validate URL
    if not url or not url.strip():
        return "[ERROR]: URL parameter is required and cannot be empty"

    try:
        # Access the URL via local RAG server
        results = await search_client.access_async([url.strip()])

        if not results:
            return "[ERROR]: No response from RAG server"

        page_content = results[0].get("page", "")

        if not page_content or page_content.strip() == "":
            return f"[ERROR]: No content found for URL: {url}\n\nThe URL may not be in the RAG server's index."

        return page_content

    except Exception as e:
        logger.error(f"Offline access error: {str(e)}")
        return f"[ERROR]: Failed to access URL '{url}': {str(e)}"


# Extraction prompt template
EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def get_prompt_with_truncation(
    info_to_extract: str, content: str, truncate_last_num_chars: int = -1
) -> str:
    """Create prompt with optional truncation."""
    if truncate_last_num_chars > 0:
        content = content[:-truncate_last_num_chars] + "[...truncated]"

    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, content)
    return prompt


async def extract_info_with_llm(
    url: str,
    content: str,
    info_to_extract: str,
    model: str = "LLM",
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    Extract information from content using an LLM API.

    Args:
        url: The source URL (for logging purposes)
        content: The content to analyze
        info_to_extract: The specific information to extract (usually a question)
        model: The model to use for extraction
        max_tokens: Maximum tokens for the response

    Returns:
        Dictionary containing:
            - success (bool): Whether the operation was successful
            - extracted_info (str): The extracted information
            - error (str): Error message if failed
            - model_used (str): The model used
            - tokens_used (int): Number of tokens used
    """
    # Validate input
    if not content or not content.strip():
        return {
            "success": False,
            "extracted_info": "",
            "error": "Content cannot be empty",
            "model_used": model,
            "tokens_used": 0,
        }

    # Validate LLM endpoint configuration
    if not SUMMARY_LLM_BASE_URL or not SUMMARY_LLM_BASE_URL.strip():
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_BASE_URL environment variable is not set",
            "model_used": model,
            "tokens_used": 0,
        }

    prompt = get_prompt_with_truncation(info_to_extract, content)

    # Prepare the payload
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0,
    }

    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if SUMMARY_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {SUMMARY_LLM_API_KEY}"

    try:
        # Retry configuration
        connect_retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(connect_retry_delays, 1):
            try:
                # Make the API request (disable proxy for internal network)
                async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
                    response = await client.post(
                        SUMMARY_LLM_BASE_URL,
                        headers=headers,
                        json=payload,
                        timeout=httpx.Timeout(None, connect=30, read=300),
                    )

                    # Check for repetition
                    if response.text and len(response.text) >= 50:
                        tail_50 = response.text[-50:]
                        repeat_count = response.text.count(tail_50)
                        if repeat_count > 5:
                            logger.info("Repeat detected in extract_info_with_llm")
                            continue

                # Check for context length errors
                if (
                    "Requested token count exceeds the model's maximum context length"
                    in response.text
                    or "longer than the model's context length" in response.text
                ):
                    prompt = get_prompt_with_truncation(
                        info_to_extract,
                        content,
                        truncate_last_num_chars=40960 * attempt,
                    )
                    payload["messages"][0]["content"] = prompt
                    continue

                response.raise_for_status()
                break  # Success, exit retry loop

            except httpx.ConnectTimeout:
                if attempt < len(connect_retry_delays):
                    logger.info(
                        f"Offline RAG Extract: Connection timeout, retry in {delay}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

            except httpx.ConnectError as e:
                if attempt < len(connect_retry_delays):
                    logger.info(
                        f"Offline RAG Extract: Connection error: {e}, retry in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

            except httpx.ReadTimeout:
                if attempt < len(connect_retry_delays):
                    logger.info(
                        f"Offline RAG Extract: Read timeout on attempt {attempt}, retry in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Offline RAG Extract: Read timeout exhausted after {len(connect_retry_delays)} attempts, url: {url}"
                    )
                    raise

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                should_retry = status_code >= 500 or status_code in [408, 409, 425, 429]

                if should_retry and attempt < len(connect_retry_delays):
                    logger.info(
                        f"Offline RAG Extract: HTTP {status_code} (retryable), retry in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise

    except Exception as e:
        error_msg = f"Offline RAG Extract: Unexpected error during LLM API call: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }

    # Parse the response
    try:
        response_data = response.json()
    except json.JSONDecodeError as e:
        error_msg = f"Offline RAG Extract: Failed to parse LLM API response: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }

    # Extract summary from response
    if "choices" in response_data and len(response_data["choices"]) > 0:
        try:
            summary = response_data["choices"][0]["message"]["content"]
        except Exception as e:
            error_msg = f"Offline RAG Extract: Failed to get summary from response: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "extracted_info": "",
                "error": error_msg,
                "model_used": model,
                "tokens_used": 0,
            }

        # Extract token usage if available
        tokens_used = 0
        if "usage" in response_data:
            tokens_used = response_data["usage"].get("total_tokens", 0)

        return {
            "success": True,
            "extracted_info": summary,
            "error": "",
            "model_used": model,
            "tokens_used": tokens_used,
        }
    elif "error" in response_data:
        error_msg = f"Offline RAG Extract: LLM API error: {response_data['error']}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }
    else:
        error_msg = "Offline RAG Extract: Unexpected LLM API response format"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": model,
            "tokens_used": 0,
        }


@mcp.tool()
async def offline_access_and_extract(url: str, info_to_extract: str) -> str:
    """
    Access webpage content from local RAG server and extract specific information using LLM.

    This tool combines offline_access with LLM-based information extraction to provide
    concise, relevant information instead of returning the full page content.

    Args:
        url: The URL to access (must be in the RAG server's index)
        info_to_extract: The specific information to extract (usually a question)

    Returns:
        JSON string containing:
            - success: Whether the operation was successful
            - url: The original URL
            - extracted_info: The extracted information
            - error: Error message if failed
            - tokens_used: Number of tokens used for extraction
    """
    # Check for RAG server address
    if not RAG_SERVER_ADDR:
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": "RAG_SERVER_ADDR environment variable not set",
                "tokens_used": 0,
            },
            ensure_ascii=False,
        )

    # Validate inputs
    if not url or not url.strip():
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": "URL parameter is required and cannot be empty",
                "tokens_used": 0,
            },
            ensure_ascii=False,
        )

    if not info_to_extract or not info_to_extract.strip():
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": "info_to_extract parameter is required and cannot be empty",
                "tokens_used": 0,
            },
            ensure_ascii=False,
        )

    try:
        # First, access the URL via local RAG server
        results = await search_client.access_async([url.strip()])

        if not results:
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "error": "No response from RAG server",
                    "tokens_used": 0,
                },
                ensure_ascii=False,
            )

        page_content = results[0].get("page", "")

        if not page_content or page_content.strip() == "":
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "error": f"No content found for URL: {url}. The URL may not be in the RAG server's index.",
                    "tokens_used": 0,
                },
                ensure_ascii=False,
            )

        # Truncate content to avoid long LLM processing time
        # Max 20k characters (~5k-7k tokens) to avoid read timeout
        max_chars = 20000
        content_truncated = False
        original_length = len(page_content)

        if len(page_content) > max_chars:
            page_content = page_content[:max_chars]
            content_truncated = True
            logger.info(
                f"Offline RAG Extract: Content truncated from {original_length} to {max_chars} chars for URL: {url}"
            )

        # Then, extract information using LLM
        extracted_result = await extract_info_with_llm(
            url=url,
            content=page_content,
            info_to_extract=info_to_extract,
            model=SUMMARY_LLM_MODEL_NAME,
            max_tokens=8192,
        )

        # Return the combined result
        return json.dumps(
            {
                "success": extracted_result["success"],
                "url": url,
                "extracted_info": extracted_result["extracted_info"],
                "error": extracted_result["error"],
                "model_used": extracted_result["model_used"],
                "tokens_used": extracted_result["tokens_used"],
            },
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"Offline access and extract error: {str(e)}")
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": f"Unexpected error: {str(e)}",
                "tokens_used": 0,
            },
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
