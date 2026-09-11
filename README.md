# BZ Studio

顕微鏡の広範囲画像、ウェルプレート画像、`.ktf`画像を表示・Stitching・書き出しするmacOS / Windows用アプリです。

## できること

- **通常画像セット** — `.gci`とOME-TIFFを使った広範囲Stitching、位置合わせ、チャンネル調整、スケールバー表示。
- **プレート画像セット** — ウェルごとの生画像タイルをStitchingして表示・調整。Zスタックは最大値／平均／中央スライスの投影、**全焦点合成（フルフォーカス）**、または**スライスのまま**（全枚、あるいはN枚ごとに1枚）で書き出せます。書き出す画像は OME-TIFF、チャンネルごとの TIFF / PNG（分離）、マージ PNG、PDF から組み合わせて選べます。
- **`.ktf`画像** — プレート表示、疑似カラー、Conditions編集、全ウェルの一括書き出し。
- **書き出し** — OME-TIFF、PNG、TIFF、PDF。別撮影のStack、time point、または両方を、順序や名称を編集して1つのPDFにできます。

## 使い方

1. [Releases](https://github.com/yoshi-koba-lab/bz-studio/releases)からmacOSまたはWindows版を取得して起動します。
2. 起動画面で「通常画像セット」「プレート画像セット」「`.ktf`」のいずれかを選び、撮影フォルダ（`.ktf`はファイル）を開きます。
3. 通常画像セットでは位置合わせ方法を確認して **Stitching実行**、プレート画像では対象ウェル、`.ktf`ではプレート上のウェルを選びます。
4. チャンネル、Conditions、スケールバー、表示倍率を調整し、形式と出力品質を選んで保存します。Stack / time series PDFは撮影フォルダを追加し、収録順を整えてから保存します。

スクロール／ピンチでズーム、ドラッグで移動、`Ctrl/Cmd+0`で全体表示します。ライセンスは[LICENSE](LICENSE)、第三者ライセンスは[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
