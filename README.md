# Deep Agents Human in the Loop example

## 準備

`.env`ファイルを作成し、以下の環境変数を設定してください。

```
ANTHROPIC_API_KEY=your-api-key-here
```

## 実行方法

### Streamlit UI

以下のコマンドでStreamlitアプリケーションを起動します。

```bash
make streamlit
```

### CLI

CLIで実行する場合は、`--input`パラメータでJSONを渡します。

```bash
uv run python cli.py --input '{"thread_id": "session-1", "type": "message", "message": "タスクの内容"}'
```

JSONの形式（`thread_id`は必須）：
- メッセージを送信する場合：`{"thread_id": "...", "type": "message", "message": "..."}`
- 承認を送信する場合：`{"thread_id": "...", "type": "approval"}`

例：
```bash
# メッセージを送信
uv run python cli.py --input '{"thread_id": "session-1", "type": "message", "message": "poem.txtにポエム書いて"}'

# 承認を送信
uv run python cli.py --input '{"thread_id": "session-1", "type": "approval"}'
```
