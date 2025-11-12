from typing import Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agent import Agent, create_my_agent, get_messages


def show_message(message: BaseMessage) -> None:
    if isinstance(message, HumanMessage):
        # ユーザーの入力の場合、そのまま表示する
        with st.chat_message(message.type):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        if len(message.tool_calls) == 0:
            # Function callingが選択されなかった場合、メッセージを表示する
            with st.chat_message(message.type):
                st.write(message.content)
        else:
            # Function callingが選択された場合、ツール名と引数を表示する
            for tool_call in message.tool_calls:
                with st.chat_message(message.type):
                    st.write(
                        f"'{tool_call['name']}' を {tool_call['args']} で実行します",
                    )
    elif isinstance(message, ToolMessage):
        # ツールの実行結果を折りたたんで表示する
        with st.chat_message(message.type):  # noqa: SIM117
            with st.expander(label="ツールの実行結果"):
                st.write(message.content)
    else:
        raise ValueError(f"Unknown message type: {message}")


def show_updates_chunk(chunk: dict[str, Any] | Any) -> None:
    if "model" in chunk:
        messages = chunk["model"]["messages"]
        for m in messages:
            show_message(m)
    if "tools" in chunk:
        messages = chunk["tools"]["messages"]
        for m in messages:
            show_message(m)


class UIState:
    def __init__(self) -> None:
        self.agent: Agent = create_my_agent()
        self.new_thread()

    def new_thread(self) -> None:
        self.thread_id = uuid4().hex


def app() -> None:
    load_dotenv(override=True)

    # UIStateを初期化
    if "ui_state" not in st.session_state:
        st.session_state.ui_state = UIState()
    ui_state: UIState = st.session_state.ui_state

    with st.sidebar:
        # 新規スレッドボタン
        clicked = st.button("新規スレッド")
        if clicked:
            ui_state.new_thread()
            st.rerun()

    st.title("Agent")
    st.write(f"thread_id: {ui_state.thread_id}")

    # 会話履歴を表示
    for m in get_messages(ui_state.agent, ui_state.thread_id):
        show_message(m)

    config: RunnableConfig = {"configurable": {"thread_id": ui_state.thread_id}}
    state = ui_state.agent.get_state(config=config)

    # interrupt中ではない場合
    if not state.next:
        # ユーザーの入力を受け付ける
        human_input = st.chat_input()
        if not human_input:
            return

        # ユーザーの入力を表示
        with st.chat_message("human"):
            st.write(human_input)

        # エージェントを実行
        for chunk in ui_state.agent.stream(
            input={"messages": [HumanMessage(content=human_input)]},
            config=config,
            stream_mode="updates",
        ):
            show_updates_chunk(chunk)

        # interruptされた可能性があるので再描画
        st.rerun()

    # interrupt中の場合
    else:
        interrupts = state.tasks[0].interrupts[0].value
        action_requests = interrupts["action_requests"]
        action_count = len(action_requests)

        # ユーザーの入力を受け付ける
        approved = st.button("承認")
        human_input = st.chat_input()

        if not approved and not human_input:
            # どちらも入力されていない場合は何もしない
            return

        elif approved:
            # action_requestsの数だけ承認を行う
            decisions = [{"type": "approve"}] * action_count
            command: Command[tuple[()]] = Command(resume={"decisions": decisions})

        elif human_input:
            # ユーザーの入力を表示
            with st.chat_message("human"):
                st.write(human_input)

            # action_requestsの数だけ拒否を行う
            decisions = [{"type": "reject", "message": human_input}] * action_count
            command = Command(resume={"decisions": decisions})

        else:
            raise ValueError(
                f"Invalid input: approved={approved}, human_input={human_input}",
            )

        for s in ui_state.agent.stream(
            input=command,
            config=config,
            stream_mode="updates",
        ):
            show_updates_chunk(s)

        # 再描画
        st.rerun()


app()
