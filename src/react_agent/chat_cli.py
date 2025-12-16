"""React Agent CLI - Simple terminal chat interface.

Purpose:
    Interactive chat loop for testing the react agent.
    Simpler than deep_research_agent CLI since we don't have
    clarification phases, supervisors, or multi-stage output.
    
Features:
    - Colored output (ANSI codes) for readability
    - Streams tool calls and final responses
    - Type 'exit' or Ctrl+C to quit
    
Usage:
    uv run python -m src.react_agent.chat_cli
"""

import sys
import asyncio
import os
from typing import List, Any, Sequence, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig

from src.react_agent.graph import react_agent
from src.react_agent.configuration import Configuration

# Simple ANSI colors with auto-disable if not a TTY
SUPPORTS_COLOR = sys.stdout.isatty()
COLOR_RESET = "\033[0m" if SUPPORTS_COLOR else ""
COLOR_INFO = "\033[36m" if SUPPORTS_COLOR else ""       # cyan
COLOR_TOOL = "\033[33m" if SUPPORTS_COLOR else ""       # yellow
COLOR_SUCCESS = "\033[32m" if SUPPORTS_COLOR else ""    # green
COLOR_PROMPT = "\033[94m" if SUPPORTS_COLOR else ""     # blue


def _truncate(text: str, limit: int = 600) -> str:
    """Truncate long tool outputs for terminal readability."""
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[truncated]..."


def _parse_stream_mode(value: str | None) -> Union[str, Sequence[str]]:
    """Parse REACT_AGENT_STREAM_MODE env var.

    Examples:
        - unset -> ("updates", "values")
        - "values" -> "values"
        - "updates,values" -> ("updates", "values")
    """
    if not value:
        return ("updates", "values")
    raw = value.strip()
    if "," not in raw:
        return raw
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(parts) if parts else ("updates", "values")


async def run_turn(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Run one turn through the agent and return updated messages."""
    print(f"{COLOR_INFO}[Agent thinking...]{COLOR_RESET}")
    
    # We'll stream full state values so we can reliably access the growing `messages` list.
    # (Default streaming yields per-node updates like {"model": ...} which won't include messages.)
    stream_mode = _parse_stream_mode(os.environ.get("REACT_AGENT_STREAM_MODE"))

    final_response: Any = None
    cfg = Configuration()
    run_config = RunnableConfig(recursion_limit=cfg.max_iterations)

    last_seen = 0
    latest_messages: List[BaseMessage] = messages
    last_step: str | None = None
    try:
        async for chunk in react_agent.astream(
            {"messages": messages},
            config=run_config,
            stream_mode=stream_mode,
        ):
            # stream_mode="values" yields {"messages": [...]}
            if isinstance(chunk, dict) and "messages" in chunk:
                chunk_messages = chunk.get("messages") or []
                latest_messages = chunk_messages

                # Only process new messages appended since last chunk
                for msg in chunk_messages[last_seen:]:
                    if isinstance(msg, ToolMessage):
                        print(f"{COLOR_TOOL}[Tool result]{COLOR_RESET} {_truncate(str(msg.content))}")
                    elif isinstance(msg, AIMessage):
                        if getattr(msg, "tool_calls", None):
                            tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                            print(f"{COLOR_TOOL}[Using tools: {', '.join(tool_names)}]{COLOR_RESET}")
                            # Helpful when debugging: show tool args (compact)
                            for tc in msg.tool_calls:
                                name = tc.get("name", "unknown")
                                args = tc.get("args", {})
                                if args:
                                    print(f"{COLOR_TOOL}  - {name} args{COLOR_RESET} {args}")
                        elif msg.content:
                            final_response = msg.content
                            print(f"{COLOR_SUCCESS}[Response]{COLOR_RESET} {final_response}")
                            print("-" * 60)

                last_seen = len(chunk_messages)

            # stream_mode="updates" yields per-node updates like {"model": ...} / {"tools": ...}
            # This is useful for step annotations (similar to LangSmith waterfall).
            elif isinstance(chunk, dict):
                # Usually single-key dicts: {"model": ...} or {"tools": ...}
                node_names = ", ".join(chunk.keys())
                if node_names and node_names != last_step:
                    print(f"{COLOR_INFO}[Step]{COLOR_RESET} {node_names}")
                    last_step = node_names
            else:
                # Unexpected streaming payload
                print(f"{COLOR_INFO}[Stream]{COLOR_RESET} {type(chunk)}")
    except GraphRecursionError as e:
        print(
            f"{COLOR_INFO}Error: {e}{COLOR_RESET}\n"
            f"{COLOR_INFO}Tip: The agent hit its iteration limit. Try rephrasing, or increase REACT_AGENT_MAX_ITERATIONS.{COLOR_RESET}"
        )
        return messages
    
    # If we never printed a final response but we do have messages, try to print the last AI output.
    if final_response is None and latest_messages:
        for msg in reversed(latest_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content:
                final_response = msg.content
                print(f"{COLOR_SUCCESS}[Response]{COLOR_RESET} {final_response}")
                print("-" * 60)
                break

    # Persist state messages for the next turn
    return latest_messages


async def main():
    """Main chat loop."""
    load_dotenv()
    print("React Agent Chat (type 'exit' to quit)")
    print()
    
    messages: List[BaseMessage] = []
    
    try:
        while True:
            # Get user input
            user_input = input(f"{COLOR_PROMPT}You{COLOR_RESET}: ").strip()
            
            if not user_input or user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            # Add user message
            messages.append(HumanMessage(content=user_input))
            
            # Run agent turn
            messages = await run_turn(messages)
            
            print()  # extra spacing before next prompt
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n{COLOR_INFO}Error: {e}{COLOR_RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
