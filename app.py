import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent import create_my_agent


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


def app() -> None:
    load_dotenv(override=True)

    st.title("My Agent")

    # 会話履歴を初期化
    if "state_messages" not in st.session_state:
        st.session_state.state_messages = []
    state_messages: list[BaseMessage] = st.session_state.state_messages

    # 会話履歴を表示
    for m in state_messages:
        show_message(m)

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()
    if not human_message:
        return

    # ユーザーの入力を表示
    with st.chat_message("human"):
        st.write(human_message)

    # ユーザーの入力を会話履歴に追加
    state_messages.append(HumanMessage(content=human_message))

    # 応答を生成
    agent = create_my_agent()

    # エージェントを実行
    for chunk in agent.stream(
        {"messages": state_messages},
        stream_mode="updates",
    ):
        if "model" in chunk:
            messages = chunk["model"]["messages"]
            for m in messages:
                show_message(m)

        if "tools" in chunk:
            messages = chunk["tools"]["messages"]
            for m in messages:
                show_message(m)


app()
