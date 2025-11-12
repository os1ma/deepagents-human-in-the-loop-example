from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent import ActionRequest, AgentStreamChunk, MyAgent


class UIState:
    def __init__(self) -> None:
        self.agent = MyAgent()
        self.new_thread()
        self.show_approve_button = False

    def new_thread(self) -> None:
        self.thread_id = uuid4().hex


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


def handle_agent_stream_chunk(chunk: AgentStreamChunk, ui_state: UIState) -> None:
    if isinstance(chunk, ActionRequest):
        ui_state.show_approve_button = True
    else:
        show_message(chunk)


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
    for m in ui_state.agent.get_messages(ui_state.thread_id):
        show_message(m)

    # 承認ボタンを表示
    if ui_state.show_approve_button:
        approved = st.button("承認")
        # 承認されたらエージェントを実行
        if approved:
            ui_state.show_approve_button = False
            with st.spinner():
                for chunk in ui_state.agent.approve(ui_state.thread_id):
                    handle_agent_stream_chunk(chunk, ui_state)
            # 会話履歴を表示するためrerun
            st.rerun()

    # ユーザーの指示を受け付ける
    human_input = st.chat_input()
    if human_input:
        with st.spinner():
            if ui_state.show_approve_button:
                ui_state.show_approve_button = False
                for chunk in ui_state.agent.reject(human_input, ui_state.thread_id):
                    handle_agent_stream_chunk(chunk, ui_state)

            else:
                for chunk in ui_state.agent.stream(human_input, ui_state.thread_id):
                    handle_agent_stream_chunk(chunk, ui_state)

            # 会話履歴を表示するためrerun
            st.rerun()


app()
