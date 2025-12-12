# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

toSingingは、任意の話し声を歌声に変換する言語非依存の音声変換システムです。MusicXML楽譜に基づいて、入力音声のピッチとタイミングを調整し、歌声を合成します。

参考文献: A. Ito, Language Independent Speech-to-Singing-Voice Conversion. Jxiv preprint, https://doi.org/10.51094/jxiv.1902

## コマンド

```bash
# 基本的な実行
python tosinging.py -i input.wav -m input.musicxml

# オプション付き
python tosinging.py -i input.wav -m input.musicxml -o output.wav -bpm 120 -trans 0

# 依存関係のインストール
pip install -r requirements.txt
```

### オプション
- `-i`: 入力WAVファイル（必須）
- `-m`: MusicXMLファイル（必須）
- `-o`: 出力ファイル名（デフォルト: output.wav）
- `-modelfile`: VUVモデルファイル（デフォルト: hubert_sad_20ms_model.pth）
- `-modelname`: HuBERTモデル名（デフォルト: facebook/hubert-base-ls960）
- `-bpm`: テンポ指定（楽譜のテンポを上書き）
- `-trans`: 移調の半音数

## アーキテクチャ

### 処理パイプライン

1. **音声分析** (`tosinging.py: analyze()`)
   - PyWORLDを使用してスペクトル、非周期性、F0を抽出
   - サンプリングレート: 16kHz

2. **有声/無声判定** (`vuv_model.py`, `HuBERT_VUV.py`)
   - HuBERTベースの3クラス分類器（有声=1, 無声=2, 無音=0）
   - 20msフレーム単位で判定

3. **楽譜解析** (`MXML2.py`)
   - MusicXMLから音符シーケンスを生成
   - 20ms単位に時間量子化

4. **音符シーケンス生成** (`noteseq.py`)
   - 音符の減衰エンベロープを生成
   - 各音符内での時間経過を[0,1]で表現し、decay関数で重み付け

5. **アライメント** (`dtw.py`)
   - 有声/無声シーケンスと楽譜のDTW（動的時間伸縮）
   - 無声区間（uv_val=2）は伸縮しない制約付き

6. **時間伸縮** (`stretch.py`)
   - アライメント結果に基づくフレームインデックスの再マッピング
   - 無声区間は長さを保持、有声区間のみ伸縮

7. **ビブラート付加** (`vibration.py`)
   - 減衰付きバネ運動モデルでピッチ遷移を滑らかに
   - 安定区間にビブラート（5Hz, 3Hz深度）を追加

8. **合成** (`tosinging.py: synthesize()`)
   - VUV判定結果で非周期性を補正
   - PyWORLDで歌声を合成

### 主要な依存関係
- `pyworld`: 音声分析・合成
- `transformers`: HuBERTモデル
- `music21`: MusicXML解析
- `librosa`: 音声処理
- `torch`: 深層学習
