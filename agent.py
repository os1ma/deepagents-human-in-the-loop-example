import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel


class ActionRequest(BaseModel):
    """人間のアクションを求めるときに画面に渡す情報を表すモデル"""

    name: str
    args: dict[str, Any]


AgentStreamChunk = AIMessage | ToolMessage | ActionRequest
"""エージェントのストリーム出力のチャンク"""


class MyAgent:
    def __init__(self) -> None:
        database_path = Path(__file__).parent / "tmp" / "checkpoint.db"
        database_conn = sqlite3.connect(database=database_path, check_same_thread=False)
        checkpointer = SqliteSaver(database_conn)

        self.agent = create_deep_agent(
            backend=FilesystemBackend(root_dir=".", virtual_mode=True),
            checkpointer=checkpointer,
            interrupt_on={
                "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
                "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            },
        )

    def stream(self, message: str, thread_id: str) -> Iterator[AgentStreamChunk]:
        stream_input = {"messages": [HumanMessage(content=message)]}
        return self._stream(stream_input, thread_id)

    def _stream(self, stream_input: Any, thread_id: str) -> Iterator[AgentStreamChunk]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for chunk in self.agent.stream(
            input=stream_input,
            config=config,
        ):
            if "model" in chunk:
                messages = chunk["model"]["messages"]
                for m in messages:
                    yield m
            if "tools" in chunk:
                messages = chunk["tools"]["messages"]
                for m in messages:
                    yield m

        state = self.agent.get_state(config=config)
        if state.next:
            interrupts = state.tasks[0].interrupts[0].value
            action_requests = interrupts["action_requests"]
            for action_request in action_requests:
                yield ActionRequest(
                    name=action_request["name"],
                    args=action_request["args"],
                )

    def approve(self, thread_id: str) -> Iterator[AgentStreamChunk]:
        decisions = [{"type": "approve"}]
        command: Command[tuple[()]] = Command(resume={"decisions": decisions})
        return self._stream(command, thread_id)

    def reject(self, feedback: str, thread_id: str) -> Iterator[AgentStreamChunk]:
        message = f"Rejected. Human feedback: {feedback}"
        decisions = [{"type": "reject", "message": message}]
        command: Command[tuple[()]] = Command(resume={"decisions": decisions})
        return self._stream(command, thread_id)

    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.agent.get_state(config=config)

        if "messages" in state_snapshot.values:
            return state_snapshot.values["messages"]  # type: ignore[no-any-return]
        else:
            return []

    def is_interrupted(self, thread_id: str) -> bool:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state = self.agent.get_state(config=config)
        return bool(state.next)
