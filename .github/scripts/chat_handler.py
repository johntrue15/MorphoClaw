#!/usr/bin/env python3
"""
Chat handler for processing OpenAI API requests with MorphoSource integration.

Search and media-detail calls are delegated to :mod:`morphosource_client`
so counts come from API pagination metadata, not ``len(items)``.
"""

import json
import os
import sys
from typing import Any, Dict, List

import requests
from openai import OpenAI

from _helpers import MORPHOSOURCE_API_BASE, get_openai_model
from morphosource_client import get_client as _get_ms_client

# Conservative token budget for requests (leave headroom for responses)
MAX_CONTEXT_TOKENS = 6000
# Maximum number of characters to keep from tool outputs to avoid huge payloads
MAX_TOOL_CONTENT_CHARS = 4000


def _normalise_content(message: Dict[str, Any]) -> str:
    """Return a string representation of message content for token estimation."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Chat Completions API may return a list of content blocks
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "type" in item:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _message_token_cost(message: Dict[str, Any]) -> int:
    base = 4  # rough overhead per message
    return base + _estimate_tokens(_normalise_content(message))


def _ensure_system_message(original: List[Dict[str, Any]], trimmed: List[Dict[str, Any]]) -> None:
    """Ensure the first system message from the conversation is preserved."""
    for msg in original:
        if msg.get("role") == "system":
            if not trimmed or trimmed[0] is not msg:
                # Insert a copy to avoid mutating the original message
                trimmed.insert(0, dict(msg))
            return


def _trim_messages(messages: List[Dict[str, Any]], max_tokens: int = MAX_CONTEXT_TOKENS) -> List[Dict[str, Any]]:
    """Return a trimmed view of messages that fits within the token budget."""
    if not messages:
        return []

    trimmed: List[Dict[str, Any]] = []
    running_total = 0

    # Iterate from the most recent message backwards so we keep the latest context
    for message in reversed(messages):
        cost = _message_token_cost(message)
        if trimmed and running_total + cost > max_tokens:
            break
        trimmed.append(message)
        running_total += cost

    trimmed.reverse()
    _ensure_system_message(messages, trimmed)

    # If adding the system message caused us to exceed the limit, drop oldest non-system messages
    while len(trimmed) > 1 and sum(_message_token_cost(msg) for msg in trimmed) > max_tokens:
        # Remove the second message (preserve the system message at index 0)
        trimmed.pop(1)

    return trimmed


def _truncate_tool_content(content: str) -> str:
    if len(content) <= MAX_TOOL_CONTENT_CHARS:
        return content
    return content[:MAX_TOOL_CONTENT_CHARS] + "... [truncated]"


def search_morphosource(query: str) -> Dict[str, Any]:
    """Search for specimens in MorphoSource database.

    Delegates to :class:`morphosource_client.MorphoSourceClient` so that
    counts reflect API pagination metadata rather than page item length.
    """
    try:
        client = _get_ms_client()
        resp = client.search_media(q=query, per_page=10)
        if resp.error:
            return {"error": resp.error, "message": resp.error}
        result = resp.raw_response or {}
        result["_total_count"] = resp.total_count
        result["_returned_count"] = resp.returned_count
        return result

    except Exception as exc:  # pragma: no cover - network errors are logged, not raised
        return {"error": str(exc)}


def get_morphosource_media(media_id: str) -> Dict[str, Any]:
    """Get details for a specific media item.

    Delegates to :class:`morphosource_client.MorphoSourceClient`.
    """
    try:
        client = _get_ms_client()
        record = client.get_media(media_id)
        if record.error:
            return {"error": record.error, "message": record.error}
        return record.data

    except Exception as exc:  # pragma: no cover - network errors are logged, not raised
        return {"error": str(exc)}


# Define tools for OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_morphosource",
            "description": "Search for specimens in the MorphoSource database by taxonomy, specimen name, or other criteria. Returns information about specimens including taxonomy, descriptions, and media URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'lizards', 'Anolis', 'crocodiles', 'CT scans')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_morphosource_media",
            "description": "Get detailed information about a specific media item from MorphoSource including voxel spacing, file formats, and specimen details",
            "parameters": {
                "type": "object",
                "properties": {
                    "media_id": {
                        "type": "string",
                        "description": "The MorphoSource media ID (e.g., '000407755')",
                    }
                },
                "required": ["media_id"],
            },
        },
    },
]


def _call_openai(client: OpenAI, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
    """Call the OpenAI chat completions API with trimmed messages."""
    trimmed_messages = _trim_messages(messages)
    return client.chat.completions.create(messages=trimmed_messages, **kwargs)


def process_chat(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process chat messages using OpenAI API with MorphoSource tools."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {"error": "OPENAI_API_KEY not configured"}

    client = OpenAI(api_key=api_key)
    conversation: List[Dict[str, Any]] = list(messages)

    try:
        _model = get_openai_model()
        response = _call_openai(
            client,
            conversation,
            model=_model,
            tools=TOOLS,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls or []

        if not tool_calls:
            return {
                "role": "assistant",
                "content": response_message.content,
            }

        conversation.append(response_message.model_dump(exclude_none=True))

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "search_morphosource":
                function_response = search_morphosource(function_args["query"])
            elif function_name == "get_morphosource_media":
                function_response = get_morphosource_media(function_args["media_id"])
            else:
                function_response = {"error": f"Unknown function: {function_name}"}

            tool_content = _truncate_tool_content(json.dumps(function_response))
            conversation.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_content,
                }
            )

        second_response = _call_openai(
            client,
            conversation,
            model=_model,
        )

        return {
            "role": "assistant",
            "content": second_response.choices[0].message.content,
            "tool_calls": [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in tool_calls
            ],
        }

    except Exception as exc:  # pragma: no cover - network errors are logged, not raised
        return {"error": str(exc)}


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: chat_handler.py '<json_payload>'")
        sys.exit(1)

    try:
        payload = json.loads(sys.argv[1])
        messages = payload.get('messages', [])

        if not messages:
            print(json.dumps({"error": "No messages provided"}))
            sys.exit(1)

        result = process_chat(messages)
        print(json.dumps(result, indent=2))

    except Exception as exc:  # pragma: no cover - just log the error
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
