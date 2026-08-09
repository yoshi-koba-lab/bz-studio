# BZ Studio

顕微鏡の広範囲画像、ウェルプレート画像、`.ktf`画像を表示・貼り合わせ・書き出す
macOS / Windows用デスクトップアプリです。

## 3つの入口

起動時に撮影形式を選びます。あとから **File ▸ ワークフローを選ぶ…** で切り替えられます。

1. **通常画像セットを読み込む**
   組織切片などのXY撮影を、`.gci`の格子・Stage座標と画像の重なりから貼り合わせます。
2. **プレート画像セットを読み込む**
   `<ウェル>/X###Y###/`構造の生画像タイルをウェルごとに貼り合わせます。
3. **`.ktf`ファイルを読み込む**
   貼り合わせ済みのプレート画像を表示し、チャンネル調整や書き出しを行います。

フォルダは撮影フォルダそのもの、または複数撮影を含む親フォルダを選べます。前回の場所は
入口ごとに記憶されます。

## 通常画像セット

`.gci`を直接含む撮影フォルダと、そこから参照される元のOME-TIFFを読み込みます。

- GCIの行・列とStage座標を初期配置に使い、隣接画像の重なりでサブピクセル精密化します。
- すべてのチャンネルに同じ幾何を適用し、チャンネル間の位置関係を保持します。
- **滑らか**は表示用の継ぎ目補正とフェザーブレンド、**最近傍**は元強度を優先する
  定量向け表示です。
- 位置合わせに使えた継ぎ目数、残差、出力寸法を画面とQCファイルで確認できます。
- 全体プレビューは軽量に保ち、拡大した領域だけ高解像度で読み直します。

### 書き出し

- **定量用 OME-TIFF** — 全チャンネル・全解像度、タイル化BigTIFF、ピラミッド付き。
  最近傍・輝度補正なしで元のDN値を保持し、スケールバーは描き込みません。
  位置合わせQCのJSONとCSVも保存します。
- **PNG / 合成TIFF / PDF** — 画面の色・min/max・ガンマを反映した合成画像。
  長辺4,000 / 8,000 / 12,000 pxから選べます。
- **スケールバー** — 自動または任意長、4隅、色、線幅、文字サイズ、余白、背景、
  任意ラベル、DPIを編集できます。

既存ファイルを置き換える場合は、実際に変更される全ファイルを実行前に表示して確認します。
書き出し途中に失敗しても、既存の正常ファイルは保持されます。

## プレート画像と`.ktf`

- ウェルプレート表示、マルチチャンネル疑似カラー、Solo、min/max、ガンマ。
- ズーム時の高解像度表示、実寸スケールバー、画素・Stage座標・強度の読取。
- Conditions表へのExcel貼り付け、列の追加・名称変更・削除、撮影ごとの自動保存。
- PNG、フル解像度TIFF、全ウェルTIFF、プレートPDF。
- **Stack / time series PDF** — 別撮影フォルダを1件ずつ追加し、撮影名・time point・
  Stack・収録順を編集して1つのPDFにまとめられます。各撮影のConditionsも個別に編集できます。

## 基本操作

- スクロール／ピンチ: ズーム
- ドラッグ: 移動
- `Ctrl/Cmd+0`: 全体表示
- `Ctrl/Cmd+1`: 100%
- `Ctrl/Cmd+Shift+A`: 自動コントラスト

## インストール

[Releases](../../releases)から環境に合うファイルを取得します。

- **macOS 15以降** — `BZ-Studio-macOS-appleSilicon.zip`または`BZ-Studio-macOS-intel.zip`を
  展開し、`BZ Studio.app`をApplicationsへ移動します。初回は右クリックして「開く」。
- **Windows** — `BZ-Studio-Windows.zip`を展開し、`BZ Studio.exe`を実行します。

ソースから起動する場合:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Python 3.9以上が必要です。

## ビルド

```bash
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm ktf_viewer.spec
```

`v*`タグではGitHub ActionsがmacOS Apple Silicon / IntelとWindows用アプリを作成します。
バージョンの正本は`version.py`です。現在 **2.0.0**。

## ライセンス

[LICENSE](LICENSE) — © 2026 yoshi-koba-lab. All Rights Reserved.

研究目的での利用、論文・発表への出力画像の使用、自分の利用範囲での改変ができます。
再配布、ミラーリング、派生物の公開、第三者へのホスティングには事前の許可が必要です。
