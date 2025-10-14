# 使い方

clone前にgit-lfsをインストールしてください

## インストール
uv をインストールしてください
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## テキスト会話の生成

ollamaをインストールし、使いたいモデル（gpt-oss:20b, gpt-oss:120b）を起動してください。  
https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama

以下でテキスト対話を生成できます
```shell
uv run generate_text_dialog.py
```

`output/transcript` にテキスト対話が生成されます

### オプション
- `--n`: 生成する対話数を指定 (デフォルト: 1)
- `--turns`: 対話の最大ターン数を指定 (デフォルト: 20)
- `--model`: 生成に使うモデル（デフォルト: gpt-oss:20b）

## 音声対話の生成

テキスト対話生成後、以下で音声対話に変換できます
```shell
uv run generate_audio_dialog.py
```

`output/audio` に音声対話が生成されます

### vitsモデルを追加する場合
`vits_model/` に config.json, style_vectors.npy と safetensorを配置してください
