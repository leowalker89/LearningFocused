"""React Agent Graph - Main LangGraph implementation.

Purpose:
    Defines a simple ReAct agent using `create_react_agent` from langgraph.prebuilt.
    The agent loops: Reason → Act (call tool) → Observe → Repeat until done.
    
Key Components:
    - create_react_agent: Pre-built LangGraph helper for ReAct pattern
    - Tools: Imported from shared tools module (Chroma, Neo4j)
    - Model: Configurable LLM (default: gpt-5-mini for speed/cost)
    
Usage:
    from src.react_agent.graph import react_agent
    result = await react_agent.ainvoke({"messages": [HumanMessage(content="...")]})
"""

# TODO: Implement react agent graph

