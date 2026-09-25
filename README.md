# whisper-local

ローカルの [faster-whisper](https://github.com/SYSTRAN/faster-whisper) を **OpenAI Audio API 互換**の HTTP サーバとして公開する。

- 既定モデル: `kotoba-tech/kotoba-whisper-v2.0-faster`（日本語特化）
- `WHISPER_MODEL` の切り替えだけで他の faster-whisper モデル（例: `large-v3-turbo`）も使える
- OpenAI SDK から `base_url=http://127.0.0.1:8000/v1` でそのまま呼べる
- GPU は自動検出。使えなければ警告ログを出して CPU にフォールバック
- word timestamps（`timestamp_granularities[]=word`）に対応

## 起動

```bash
uv run python -m server
# -> Listening on http://0.0.0.0:8000
```

`whisper-local` コンソールスクリプトも同じサーバを起動する（`uv run whisper-local`）。

モデルは起動時に1回だけロードされる（初回は数秒）。`GET /healthz` が `model_loaded: true` を返したら準備完了。

## モデルの切り替え

使用モデルは `WHISPER_MODEL` で指定する（既定は kotoba）。同時に複数モデルは扱わず、**env を変えて起動し直す**方式。

```bash
# 日本語特化（既定）
uv run python -m server

# 多言語・汎用の turbo
WHISPER_MODEL=large-v3-turbo uv run python -m server
```

`large-v3-turbo` は faster-whisper の組み込みエイリアス（実体 `mobiuslabsgmbh/faster-whisper-large-v3-turbo`）。初回は HF から取得が必要（`local_files_only` で読むため、事前取得しておくこと）。

| model | 実測速度 (CPU int8) | ピーク RSS | words | 備考 |
|---|---|---|---|---|
| `kotoba-tech/kotoba-whisper-v2.0-faster`（既定） | 約 1.8x realtime | 約 1.7 GB | 可（補正あり） | 日本語特化 |
| `large-v3-turbo` | 約 1.9x realtime（`cpu_threads=8` で 1.7x） | 約 1.7 GB | 可（**補正不要**） | 多言語・汎用 |

- 速度は「音声1秒あたりの処理秒（RTF）」。数値は 3.29s の日本語音声での実測。
- turbo は `alignment_heads`（最大層3）が実デコーダ4層に収まるため、**補正は自動でスキップされる**（起動ログに `no correction needed` と出る）。
- `large-v3-turbo` は `cpu_threads=8` を付けるとわずかに速い（`WHISPER_CPU_THREADS=8`）。

## 環境変数

| env | 既定 | 意味 |
|---|---|---|
| `WHISPER_MODEL` | `kotoba-tech/kotoba-whisper-v2.0-faster` | 使用モデル（HF ID またはローカルディレクトリ） |
| `WHISPER_DEVICE` | `auto` | `auto` / `cpu` / `cuda`。`auto` は CUDA 利用可否を実測して決定 |
| `WHISPER_COMPUTE_TYPE` | device 依存 | `int8` / `float16` / `float32`（既定: cuda→`float16`, cpu→`int8`） |
| `WHISPER_HOST` | `0.0.0.0` | bind アドレス |
| `WHISPER_PORT` | `8000` | bind ポート |
| `WHISPER_API_KEY` | 未設定 | 設定時のみ `Authorization: Bearer` を検証（未設定なら認証スキップ） |
| `WHISPER_MAX_UPLOAD_MB` | `100` | アップロード上限（超過は 413） |
| `WHISPER_CPU_THREADS` | `0`(auto) | CTranslate2 の CPU スレッド数 |
| `WHISPER_VAD_FILTER` | `false` | VAD 前段フィルタを有効化 |

例:

```bash
WHISPER_API_KEY=secret WHISPER_PORT=9000 WHISPER_MAX_UPLOAD_MB=50 uv run python -m server
```

## エンドポイント

| method | path | 説明 |
|---|---|---|
| POST | `/v1/audio/transcriptions` | 文字起こし（multipart/form-data） |
| POST | `/v1/audio/translations` | 英語への翻訳（`task=translate`） |
| GET | `/v1/models` | OpenAI クライアントの疎通確認用モデル一覧 |
| GET | `/healthz` | 死活監視（モデルロード状態・device・words 可用性） |
| GET | `/-/healthcheck/` | 起動待ちヘルスチェック（認証不要・常に `{"status":"ok"}`。スラッシュなしも 200） |
| GET | `/` | 簡易情報（バージョン・対応エンドポイント） |

### パラメータ（transcriptions / translations）

`file`（必須）, `model`, `language`, `prompt`, `response_format`, `temperature`, `timestamp_granularities[]`

- `response_format`: `json`(既定) / `text` / `srt` / `vtt` / `verbose_json`
- `temperature`: `0` のみ許可（それ以外は 400）
- `timestamp_granularities[]`: `segment`（既定）/ `word`
- `stream=true` は未対応（400）

### エラー形式

OpenAI と同じ封筒形式:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "file", "code": null}}
```

`400`（パラメータ不備 / 音声デコード不能）, `401`（認証失敗）, `413`（サイズ超過）, `500`（推論失敗）。

## curl 例

```bash
AUDIO=/path/to/audio.wav

# json（既定）
curl -s http://127.0.0.1:8000/v1/audio/transcriptions -F file=@"$AUDIO"
# -> {"text":"..."}

# text
curl -s http://127.0.0.1:8000/v1/audio/transcriptions -F file=@"$AUDIO" -F response_format=text

# srt / vtt
curl -s http://127.0.0.1:8000/v1/audio/transcriptions -F file=@"$AUDIO" -F response_format=srt
curl -s http://127.0.0.1:8000/v1/audio/transcriptions -F file=@"$AUDIO" -F response_format=vtt

# verbose_json + word timestamps
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@"$AUDIO" -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word' -F language=ja

# 翻訳
curl -s http://127.0.0.1:8000/v1/audio/translations -F file=@"$AUDIO"

# 死活確認
curl -s http://127.0.0.1:8000/healthz
```

## OpenAI SDK 例

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

print(result.text)
for word in result.words or []:
    print(word.word, word.start, word.end)
```

`api_key` はダミーでよい（`WHISPER_API_KEY` 未設定時はサーバ側で検証しない）。`openai` パッケージが無い環境では、`scripts/smoke.py`（httpx で同等の multipart 送信）が使える:

```bash
uv run python scripts/smoke.py /path/to/audio.wav
```

## テスト

```bash
uv run pytest
```

API 契約テスト（`tests/test_api.py`）はスタブの `ModelManager` を使うため、モデルロード不要で高速に回る。

## 既知の制約

- **GPU 自動検出**: 起動時に CUDA 利用可否を実測する。現環境は `libcuda.so.1` 不在のため **CPU（int8）で動作**し、警告ログが出る。CUDA ユーザースペースを導入して再起動すれば自動で GPU（float16）に乗る。`WHISPER_DEVICE=cuda` を明示指定して CUDA ロードに失敗した場合も、警告を出して CPU にフォールバックする。
- **alignment_heads 補正**: `kotoba-whisper-v2.0-faster` の配布 `config.json` は `alignment_heads` がデコーダ層 7〜25 を指すが、`model.bin` の実デコーダ層数は 2。CTranslate2 は範囲チェックせず OOB 書き込みし、word timestamps 指定時に **segfault（exit 139）**する。起動時に層数を検出し、範囲外なら最終層の全ヘッドに補正したモデルディレクトリを `<repo>/var/models/<model>/` に生成して読み込む（`model.bin` は symlink、HF キャッシュは変更しない）。補正に失敗した場合は警告を出し、words を無効化した状態で起動継続する（サーバは落とさない）。詳細: `thoughts/word-timestamps-forensics.md`
- **初回起動**: HF キャッシュにモデルが無い場合はロードできない（`local_files_only` で取得するため、事前にモデルを取得しておくこと）。
- **認証は既定で無効**: `0.0.0.0` bind のため、LAN 内から誰でも叩ける。必要な場合は `WHISPER_API_KEY` を設定する。
- **ストリーミング未対応**: `stream=true` は 400 を返す。
- **`include[]=logprobs` 未対応**: 受け取るが無視する。
- **`verbose_json` の `words`** は `timestamp_granularities[]=word` 指定時のみ埋まる（未指定なら空配列）。`segments[].seek` は faster-whisper の値をそのまま返す。
- **CPU 推論速度**: 実測でおおよそ 1.8x realtime（3.29s 音声で推論 ≒6s）。

## レイアウト

```
whisper-local/
├── server/                       # `uv run python -m server` シム
├── scripts/smoke.py              # 実サーバ向けスモーク（httpx）
├── src/whisper_local/
│   ├── __init__.py               # 副作用なし・公開 API のみ
│   ├── config.py                 # env -> Settings
│   ├── alignment.py              # 層数検出 + alignment_heads 補正
│   ├── transcriber.py            # モデル単一ロード / ロック直列化 / words
│   ├── schemas.py                # json / text / verbose_json 整形
│   └── server/
│       ├── app.py                # FastAPI アプリ本体
│       ├── main.py               # uvicorn 起動
│       ├── openai_types.py       # OpenAI 互換型・エラー封筒
│       └── format_srt.py         # srt / vtt 整形
└── tests/test_api.py             # API 契約テスト
```
