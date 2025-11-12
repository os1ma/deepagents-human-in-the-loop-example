from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

Agent = CompiledStateGraph[Any, None, Any, Any]


def create_my_agent() -> CompiledStateGraph[Any, None, Any, Any]:
    return create_deep_agent(  # type: ignore[no-any-return]
        backend=FilesystemBackend(root_dir=".", virtual_mode=True),
        checkpointer=MemorySaver(),
        interrupt_on={
            "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        },
    )


def get_messages(agent: Agent, thread_id: str) -> list[BaseMessage]:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state_snapshot = agent.get_state(config=config)

    if "messages" in state_snapshot.values:
        return state_snapshot.values["messages"]  # type: ignore[no-any-return]
    else:
        return []
