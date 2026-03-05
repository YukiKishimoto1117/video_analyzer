# Video Corner Analyzer

動画をアップロードすると、以下の流れでコーナー分析します。

1. Whisperで音声を文字起こし
2. Gemini APIで文字起こし内容をコーナー単位に分類
3. コーナー情報（開始秒・終了秒・タイトル・概要）をSQLiteへ保存
4. フロント画面で表形式表示

## 想定アーキテクチャ

- **Frontend**: HTML + JS + CSS（FastAPIから配信）
- **Backend API**: FastAPI
  - `POST /api/upload`: 動画アップロード、解析実行、DB保存
  - `GET /api/videos/{id}`: 保存済み結果の取得
- **DB**: SQLite（SQLAlchemy）
  - `videos` テーブル
  - `corners` テーブル
- **Transcription**: Whisper（ローカルモデル）
- **Classification**: Gemini API

## ディレクトリ構成

```text
app/
  main.py            # FastAPIエントリーポイント
  config.py          # 設定（環境変数）
  database.py        # DB接続
  models.py          # SQLAlchemyモデル
  schemas.py         # APIレスポンススキーマ
  services.py        # Whisper/Gemini呼び出し
  templates/index.html
  static/app.js
  static/style.css
  uploads/           # アップロード動画保存先
  data/              # SQLiteファイル
tests/
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
uvicorn app.main:app --reload
```

ブラウザで `http://127.0.0.1:8000` を開いて動画をアップロードしてください。

## 環境変数

- `GEMINI_API_KEY`: Gemini APIキー（未設定時はフォールバックで全体を1コーナーとして保存）
- `WHISPER_MODEL`: Whisperモデル名（例: `base`, `small`）
- `DATABASE_URL`: DB接続文字列

## 今後の改善案

- 非同期ジョブ化（Celery / RQ）で長時間処理をバックグラウンド化
- FFmpegで音声抽出を先に実行してWhisper処理時間を短縮
- Gemini出力JSONの厳格バリデーション
- ユーザー認証・履歴画面追加


## APIキー管理（重要）

Gemini APIキーは**フロント（HTML/JavaScript）に書かない**でください。  
ブラウザに配信されるコードにキーを書くと、誰でも閲覧できてしまいます。

このリポジトリでは、`app/config.py` が `.env` を読み込むため、以下の運用が安全です。

1. `.env.example` をコピーして `.env` を作成
2. `.env` の `GEMINI_API_KEY=` に本物のキーを設定
3. `.env` は `.gitignore` で除外されるため、`git add .` しても通常はコミットされない

```bash
cp .env.example .env
# .env を開いて GEMINI_API_KEY=... を設定
```

### 追加の注意点

- APIキーはサーバー側（FastAPI）だけで使用し、クライアント側には返さない。
- もしキーを誤ってコミットしたら、**即時ローテーション（再発行）**してください。
- 本番環境では `.env` ではなく、クラウドのシークレット管理（Secret Manager, Parameter Store等）を推奨します。
