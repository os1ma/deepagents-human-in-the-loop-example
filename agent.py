from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.graph.state import CompiledStateGraph


def create_my_agent() -> CompiledStateGraph:
    return create_deep_agent(
        backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    )
