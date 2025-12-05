"""React Agent Prompts - System prompts for the ReAct agent.

Purpose:
    Contains the system prompt(s) that guide the agent's behavior.
    Unlike deep_research_agent, this is a single-prompt setup.
    
Key Components:
    - system_prompt: Main instruction for the agent (domain focus, tool usage guidance)
    - Dynamically includes tool legend from utils.build_tool_legend()
    
Design Notes:
    - Keep prompts concise; ReAct agents work best with clear, direct instructions
    - Include context about the podcast knowledge base (Alpha School, Two Hour Learning)
    - Guide the agent to use tools before answering
"""

# TODO: Define system_prompt

