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

## 録音 UI（ブラウザから録音 → 即時文字起こし）

```bash
uv run python -m server
```

起動後、ブラウザで **<http://127.0.0.1:8000/ui>** を開く（`http://localhost:8000/ui` でも可）。

- 「● 録音開始」→ マイクに向かって話す →「■ 停止して認識」で、停止と同時に
  `POST /v1/audio/transcriptions`（`verbose_json` + `timestamp_granularities[]=word`, `language=ja`）へ送信する。
- 認識結果の本文・セグメント・words・音声長・処理時間を表示。結果は「結果をコピー」でクリップボードへ。
- 履歴は**セッション内のみ**（リロードで消える。`localStorage` は使わない）。
- ヘッダに `GET /healthz` の結果（モデル名・device・`model_loaded`）を表示する。
- `WHISPER_API_KEY` を設定している場合は画面の「API キー」欄に入力すると `Authorization: Bearer` で送信する。
- マイクを許可できない場合・非セキュアコンテキスト（`http://` の外部ホスト等）では案内を表示する。
  `getUserMedia` は `127.0.0.1` / `localhost` / HTTPS でのみ使える。

実装は `src/whisper_local/server/static/index.html` の1ファイル完結（素の HTML/CSS/JS、ビルド工程なし）。
`POST` の CORS は全許可（`allow_origins=["*"]`）なので、`file://` や別ポートのページからも叩ける。

### 録音音声の保存

アップロードされた音声は既定で保存される（`var/` は `.gitignore` 済み）。

```
var/recordings/<YYYY-MM-DD>/<HHMMSS>-<6桁乱数>.<ext>        # 音声本体
var/recordings/<YYYY-MM-DD>/<HHMMSS>-<6桁乱数>.json         # verbose_json の sidecar
```

- `ext` は multipart の元ファイル名の拡張子、無ければ `Content-Type` から決定（`webm` / `m4a` / `ogg` / `wav` など、不明なら `bin`）。
- sidecar JSON には `result`（`verbose_json` 全文）に加え、モデル名・`processing_seconds`・元ファイル名・`audio_url`（`/api/recordings/<date>/<name>`）を入れる。
- 保存に失敗しても認識結果は 200 のまま返る（警告ログのみ）。
- `GET /api/recordings/<date>/<name>` で保存音声を取得できる（`WHISPER_API_KEY` 設定時は要 Bearer）。

| env | 既定 | 意味 |
|---|---|---|
| `WHISPER_SAVE_AUDIO` | `true` | `false` で保存を無効化 |
| `WHISPER_SAVE_AUDIO_DIR` | `<repo>/var/recordings` | 保存先ディレクトリ（絶対パス可） |

## ストリーミング文字起こし（Deepgram 互換 `WS /v1/listen`）

Deepgram の [Live Audio API](https://developers.deepgram.com/reference/speech-to-text/listen-streaming)（`wss://api.deepgram.com/v1/listen`）互換の WebSocket を同じプロセスで提供する。
エンジンは既存の faster-whisper（`kotoba` 既定）を再利用し、**生バイナリ PCM（`encoding=linear16`）を受信 → 16 kHz mono float32 へ変換 → バッファ + 簡易エンドポインティングで interim / final を擬似ストリーム**する。

```bash
uv run python -m server
# ブラウザで http://127.0.0.1:8000/ui/listen を開く
# または CLI から:
uv run python scripts/deepgram_client.py /path/to/audio.wav --language ja --finalize
```

### 接続

- エンドポイント: `ws://<host>:<port>/v1/listen`（本PoCは自前ホスト）
- 認証（`WHISPER_API_KEY` 設定時のみ検証。3経路のいずれか）:
  - ヘッダ `Authorization: Token <KEY>`（`Bearer <KEY>` も可）
  - クエリ `?api_key=<KEY>`
  - `Sec-WebSocket-Protocol: token, <KEY>`（ブラウザ等カスタムヘッダ不可の環境向け。受理時はサブプロトコル `token` を返す）
  - 失敗時は upgrade を拒否して **HTTP 401**（`{"err_code":"INVALID_AUTH","err_msg":"Invalid credentials.","request_id":...}`）
- SDK が付与する `x-deepgram-session-id` 等の追加ヘッダは受理して無視する

### クエリパラメータ

| param | 既定 | 挙動 |
|---|---|---|
| `model` | — | 受理（ログ用。実体は起動時の faster-whisper に固定） |
| `language` | `ja` | 受理（ISO-639-1） |
| `encoding` | `linear16` | `linear16` / `mulaw` / `alaw`。別名 `pcm16` `ulaw` `g711_ulaw` `g711_alaw` も受理。他は `Error` |
| `sample_rate` | encoding 依存 | `linear16`→`16000`、`mulaw`/`alaw`→`8000`。省略時はこの既定。16k 以外は numpy でリサンプル |
| `channels` | `1` | `1` のみ。`>1` は `Error` |
| `interim_results` | `false` | `true` で interim `Results`（`is_final=false`）を送出 |
| `endpointing` | `500`(PoC内部既定) | final 判定の無音長（ms）。`false`/`0` で無効 |
| `utterance_end_ms` | 無効 | 指定で `UtteranceEnd` を送出 |
| `vad_events` | `false` | `true` で `SpeechStarted` を送出 |
| `punctuate` / `smart_format` / `version` | — | 受理（`version` は無視、他はモデル出力依存で素通し） |
| その他（`diarize` / `keywords` / `keyterm` / `redact` 等） | — | 受理して無視 |

> Deepgram の `endpointing` 既定は `10`(ms) だが、faster-whisper は VAD を持たないバッチ推論のため、極端に短い値は誤検知になる。クエリ未指定時の PoC 内部既定は **500 ms**（§4.5）。

### クライアント → サーバ

| message | 形式 | 挙動 |
|---|---|---|
| 音声 | **生バイナリフレーム**（base64 ではない） | バッファに蓄積 |
| `{"type":"Finalize"}` | JSON | 現バッファを final 化（`from_finalize=true`） |
| `{"type":"CloseStream"}` | JSON | final 化 → `Metadata` → `close(1000)` |
| `{"type":"KeepAlive"}` | JSON | 受理（無音バッファには積まない） |

### サーバ → クライアント

| message | 主なフィールド | タイミング |
|---|---|---|
| `Results` | `type, channel_index, duration, start, is_final, speech_final, from_finalize, channel.alternatives[].{transcript,confidence,words[]}, metadata` | interim / final のたび |
| `Metadata` | `type, transaction_key, request_id, sha256, created, duration, channels` | `CloseStream` の final 後 |
| `UtteranceEnd` | `type, channel, last_word_end` | `utterance_end_ms` 経過時 |
| `SpeechStarted` | `type, channel, timestamp` | 発話開始時（`vad_events=true`） |
| `Error` | `type, err_code, err_msg, request_id` | 不正クエリ / 復号不能フレーム等 |

- `is_final=false` = interim（更新されうる） / `is_final=true` = 確定 / `speech_final=true` = 発話の切れ目（エンドポインティング確定） / `from_finalize=true` = `Finalize`/`CloseStream` 起因。
- **無音のみの final は送出しない**（ハルシネーション回避）。
- `words[]` は word timestamps 有効時（`/healthz` の `words_available`）のみ埋まる。無効なら `transcript`/`confidence` のみ。
- `confidence` は words があれば word 確率の平均、無ければ segment の `avg_logprob` から `exp()` で算出。

### ブラウザ UI（`/ui/listen`）

- マイクを `AudioContext` で取得し、**`ScriptProcessorNode` で 4096 フレームずつ Int16 PCM に変換して送信**。
- 暫定結果は薄字、確定結果は太字で逐次表示（Deepgram 流）。`SpeechStarted` / `UtteranceEnd` / `Metadata` はログ欄に表示。
- `Finalize` は使わず、停止時に `CloseStream` を送って最終結果を待つ。30 秒以内の無音で切断されないよう 8 秒ごとに `KeepAlive` を送る。
- 言語・エンドポインティング・`UtteranceEnd`・API キー・interim / `vad_events` の ON/OFF をフォームで指定できる。
- `index.html` と同じ流儀の1ファイル完結（素の HTML/CSS/JS、ビルド工程なし）。

### デバッグ用 CLI

```bash
uv run python scripts/deepgram_client.py /path/to/audio.wav \
  --language ja --interim-results --vad-events --utterance-end-ms 1000

# G.711 µ-law / 8 kHz（OpenClaw の Dictation relay と同じ形式）
uv run python scripts/deepgram_client.py /path/to/audio.wav --encoding g711_ulaw
```

wav（PyAV が開ける形式）を `--encoding` に応じた形式（既定 `linear16`/16 kHz、`mulaw`/`alaw` は ffmpeg で G.711 へ圧縮し 8 kHz）に変換して 0.5 秒フレームで送信し、サーバメッセージを生ログで表示する。`--api-key` で `Authorization: Token`、`--finalize` で `Finalize` を送る。

### 制限

- 真のトークン逐次デコードは無い（faster-whisper 非対応）。あくまで「バッファ + エンドポインティング」による擬似ストリーム。
- CPU 推論は実時間より遅いため interim は既定で控えめ（`WHISPER_LISTEN_INTERIM_INTERVAL_MS`）。`interim_results=false` でも final は成立する。
- 推論は `ModelManager._lock` で HTTP と直列化される（同時接続時は待ちが発生）。
- 1 チャンネルのみ。`encoding` は `linear16` / `mulaw` / `alaw`（別名 `g711_ulaw` `g711_alaw`）。G.711 は 8 kHz・1 サンプル 1 バイトで、Python 3.13 で削除された stdlib `audioop` の代わりに numpy 実装（ffmpeg のデコード結果と全 256 値一致を確認済み）。コンテナ入力（webm/opus 等）は未対応（`Error`）。

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
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `false` | 前セグメント出力を次セグメントの prompt に引き継ぐ（**既定 `false`。`true` は kotoba モデルで長尺が打ち切られる**） |
| `WHISPER_SAVE_AUDIO` | `true` | アップロード音声の保存（`false` で無効化） |
| `WHISPER_SAVE_AUDIO_DIR` | `<repo>/var/recordings` | 音声の保存先 |
| `WHISPER_LISTEN_INTERIM_INTERVAL_MS` | `1500` | `/v1/listen` の interim 送信間隔（ms）。`interim_results=true` 時のみ有効 |

例:

```bash
WHISPER_API_KEY=secret WHISPER_PORT=9000 WHISPER_MAX_UPLOAD_MB=50 uv run python -m server
```

## エンドポイント

| method | path | 説明 |
|---|---|---|
| POST | `/v1/audio/transcriptions` | 文字起こし（multipart/form-data） |
| POST | `/v1/audio/translations` | 英語への翻訳（`task=translate`） |
| WS | `/v1/listen` | Deepgram 互換リアルタイム文字起こし（生 `linear16` PCM → `Results`/`Metadata`/`UtteranceEnd`/`SpeechStarted`） |
| GET | `/v1/models` | OpenAI クライアントの疎通確認用モデル一覧 |
| GET | `/healthz` | 死活監視（モデルロード状態・device・words 可用性） |
| GET | `/ui` | 録音 UI（HTML。`/ui/` も 200） |
| GET | `/ui/listen` | ストリーミング文字起こし UI（HTML） |
| GET | `/api/recordings/<date>/<name>` | 保存済み録音音声の取得 |
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
- **長尺音声の打ち切り（既定 `WHISPER_CONDITION_ON_PREVIOUS_TEXT=false` で回避済み）**: faster-whisper の既定 `condition_on_previous_text=True` は、前セグメントの出力トークンを次セグメントの `prompt` に引き継ぐ。kotoba モデル＋長尺入力ではこれが同一トークン列の繰り返しに陥り、Whisper が「これ以上テキストは無い」と判断して**残り時間をスキップして早期終了する**（20 秒台前半で打ち切られる）。このため本サーバは既定を `false` にしている。実測（日本語音声）:

  | 音声長 | `true`（faster-whisper 既定） | `false`（本サーバ既定） |
  |---|---|---|
  | 42 秒 | 22.62 / 42.02 秒 = **54%** | 41.06 / 42.02 秒 = **98%** |
  | 191 秒 | 23.58 / 191.62 秒 = **12%** | 190.66 / 191.62 秒 = **100%** |

  `true` に戻したい場合（多言語モデルで文脈継続を試す等）は `WHISPER_CONDITION_ON_PREVIOUS_TEXT=true` で起動する。デコード・VAD・`word_timestamps`・アップロード上限はいずれも原因ではなく（実測で除外済み）、この設定のみが原因。
- **認証は既定で無効**: `0.0.0.0` bind のため、LAN 内から誰でも叩ける。必要な場合は `WHISPER_API_KEY` を設定する。
- **ストリーミング未対応**: `stream=true` は 400 を返す。
- **`include[]=logprobs` 未対応**: 受け取るが無視する。
- **`verbose_json` の `words`** は `timestamp_granularities[]=word` 指定時のみ埋まる（未指定なら空配列）。`segments[].seek` は faster-whisper の値をそのまま返す。
- **CPU 推論速度**: 実測でおおよそ 1.8x realtime（3.29s 音声で推論 ≒6s）。
- **録音 UI はマイク必須**: ブラウザの `getUserMedia` を使うため、`https` か `127.0.0.1`/`localhost` でしか動作しない。`0.0.0.0` で LAN から IP 直指定で開くと録音できない（UI 側で案内を出す）。

## 入力長の上限

- **サーバ側・モデル側に入力長の制限は無い。** 唯一の上限はアップロードサイズの `WHISPER_MAX_UPLOAD_MB`（既定 100 MB、超過は 413）。
- opus/webm を 32 kbps で送る場合、100 MB は**約 7 時間相当**。
- 処理時間（CPU int8）は約 **1.8x realtime**（1 分の音声で推論 ≒40〜60 秒）。律速はアップロード容量と待ち時間で、長尺は分割送信より一括送信の方が速い（191 秒で ≒41 秒）。
- 長尺で後半が欠落する症状は長さ制限ではなく、上記「既知の制約」の `condition_on_previous_text` が原因（既定 `false` で解決済み）。

## レイアウト

```
whisper-local/
├── server/                       # `uv run python -m server` シム
├── scripts/smoke.py              # 実サーバ向けスモーク（httpx）
├── scripts/deepgram_client.py    # WS /v1/listen のデバッグクライアント
├── scripts/deepgram_sdk_check.py # Phase 4: 公式 deepgram-python-sdk 接続検証
├── src/whisper_local/
│   ├── __init__.py               # 副作用なし・公開 API のみ
│   ├── config.py                 # env -> Settings
│   ├── alignment.py              # 層数検出 + alignment_heads 補正
│   ├── audio.py                  # linear16 -> 16k float32 / リサンプル / RMS
│   ├── transcriber.py            # モデル単一ロード / ロック直列化 / words / transcribe_stream
│   ├── schemas.py                # json / text / verbose_json 整形
│   └── server/
│       ├── app.py                # FastAPI アプリ本体 / WS /v1/listen
│       ├── listen.py             # Deepgram 互換プロトコル（状態/VAD/イベント）
│       ├── main.py               # uvicorn 起動
│       ├── openai_types.py       # OpenAI 互換型・エラー封筒
│       ├── static/index.html     # 録音 UI（1ファイル完結）
│       ├── static/listen.html    # ストリーミング文字起こし UI（1ファイル完結）
│       └── format_srt.py         # srt / vtt 整形
├── tests/test_api.py             # API 契約テスト
└── tests/test_listen.py          # WS /v1/listen 契約テスト
```
