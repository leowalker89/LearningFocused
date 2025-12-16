"""React Agent Graph - Main agent implementation.

Purpose:
    Defines a simple ReAct agent using `create_agent` from langchain.agents.
    The agent loops: Reason → Act (call tool) → Observe → Repeat until done.
    
Key Components:
    - create_agent: LangChain helper for an agent runtime (built on LangGraph)
    - Tools: Imported from shared tools module (Chroma, Neo4j)
    - Model: Configurable LLM (default: gpt-5-mini for speed/cost)
    
Usage:
    from src.react_agent.graph import react_agent
    result = await react_agent.ainvoke({"messages": [HumanMessage(content="...")]})
"""

from typing import Any, Optional
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.react_agent.configuration import Configuration
from src.react_agent.tools import tools
from src.react_agent.utils import create_chat_model
from src.react_agent.prompts import system_prompt


def get_react_agent(config: Optional[RunnableConfig] = None) -> Any:
    """Create a ReAct agent with the specified configuration.
    
    Args:
        config: Optional RunnableConfig with configurable dict (model, max_tokens, etc.)
        
    Returns:
        LangChain agent runnable that accepts {"messages": [...]}
    """
    cfg = Configuration.from_runnable_config(config)
    
    # Create the chat model instance
    model = create_chat_model(
        model_name=cfg.model,
        config=config,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
    )
    
    # Create agent with model and tools
    # create_agent returns a compiled graph that acts as a runnable
    agent: Any = create_agent(model, tools=tools, system_prompt=system_prompt)

    # Respect configured iteration limits; default to Configuration.max_iterations
    recursion_limit = None
    if isinstance(config, dict):
        recursion_limit = config.get("recursion_limit")
    if recursion_limit is None:
        recursion_limit = cfg.max_iterations

    return agent.with_config(recursion_limit=recursion_limit)


# Create default agent instance
react_agent = get_react_agent()
