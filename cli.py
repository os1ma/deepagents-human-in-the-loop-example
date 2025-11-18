import argparse
import json

from dotenv import load_dotenv

from agent import ActionRequest, AgentStreamChunk, MyAgent


def show_agent_stream_chunk(chunk: AgentStreamChunk) -> None:
    if isinstance(chunk, ActionRequest):
        output = {
            "type": "action_request",
            "name": chunk.name,
            "args": chunk.args,
        }
    else:
        output = {
            "type": "message",
            "data": chunk.model_dump(),
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent CLI")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            'JSON input: {"thread_id": "...", "type": "message", "message": "..."} '
            'or {"thread_id": "...", "type": "approval"} (thread_id is required)'
        ),
    )
    args = parser.parse_args()

    # 環境変数を読み込む
    load_dotenv(override=True)

    # JSON入力をパース
    input_data = json.loads(args.input)

    # 必須フィールドをチェック
    if "thread_id" not in input_data:
        raise ValueError("'thread_id' field is required in input JSON")

    if "type" not in input_data:
        raise ValueError("'type' field is required in input JSON")

    thread_id = input_data["thread_id"]
    input_type = input_data["type"]

    # エージェントを初期化
    agent = MyAgent()

    # 入力タイプに応じて処理
    if input_type == "message":
        message = input_data["message"]

        if not agent.is_interrupted(thread_id):
            for chunk in agent.stream(message, thread_id):
                show_agent_stream_chunk(chunk)

        else:
            for chunk in agent.reject(message, thread_id):
                show_agent_stream_chunk(chunk)

    elif input_type == "approval":
        for chunk in agent.approve(thread_id):
            show_agent_stream_chunk(chunk)

    else:
        raise ValueError(
            f"Unknown type '{input_type}'. Must be 'message' or 'approval'",
        )


if __name__ == "__main__":
    main()
