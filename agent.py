from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph


def create_my_agent() -> CompiledStateGraph:
    return create_deep_agent()
