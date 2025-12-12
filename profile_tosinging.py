"""
toSinging プロファイリングスクリプト
各処理ステップの実行時間を計測してボトルネックを特定する
"""
import time
import pyworld as pw
import librosa
import numpy as np
import soundfile as sf
import vuv_model as vuv
import dtw
from noteseq import noteseq
import MXML2 as mxml
import vibration
import stretch

TARGET_SR = 16000


def load_audio_fast(filepath, target_sr=TARGET_SR):
    """soundfileで読み込み後リサンプリング（librosa.loadより高速）"""
    w, sr = sf.read(filepath)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return w, target_sr


def profile_convert_speech2sing(inputfile, musicxmlfile, outputfile,
                                modelfile, modelname,
                                transpose=0, bpm=None,
                                use_cached_model=True,
                                use_fast_audio=True):
    """プロファイリング付きの変換処理"""
    timings = {}
    total_start = time.perf_counter()

    # [1] モデル読み込み
    t0 = time.perf_counter()
    if use_cached_model:
        model = vuv.get_cached_model(modelfile, modelname, 3, TARGET_SR)
    else:
        model = vuv.VUVmodel(modelfile, modelname, 3, TARGET_SR)
    timings['[1] Model load'] = time.perf_counter() - t0

    # [2] 音声ファイル読み込み
    t0 = time.perf_counter()
    if use_fast_audio:
        w, sr = load_audio_fast(inputfile, TARGET_SR)
    else:
        w, sr = librosa.load(inputfile, sr=TARGET_SR)
    timings['[2] Audio load'] = time.perf_counter() - t0

    # [3] 音声分析 (PyWORLD)
    t0 = time.perf_counter()
    yf = np.array(w, dtype=np.float64)
    _f0, t = pw.harvest(yf, sr)
    f0 = pw.stonemask(yf, _f0, t, sr)
    ap = pw.d4c(yf, f0, t, sr)
    sp = pw.cheaptrick(yf, f0, t, sr)
    anl = {"spectrum": sp, "aperiodicity": ap, "f0": f0, "rate": sr}
    timings['[3] PyWORLD analyze'] = time.perf_counter() - t0

    # [4] MusicXML解析
    t0 = time.perf_counter()
    if bpm is None:
        src_notes = mxml.musicxml_to_df(musicxmlfile)
    else:
        src_notes = mxml.musicxml_to_df(musicxmlfile, default_bpm=bpm, force_default_bpm=True)
    if transpose != 0:
        mxml.transpose_pitch(src_notes, transpose)
    timings['[4] MusicXML parse'] = time.perf_counter() - t0

    # [5] 音符シーケンス生成
    t0 = time.perf_counter()
    notes, pitches, note_start, note_end = noteseq(src_notes)
    timings['[5] noteseq'] = time.perf_counter() - t0

    # [6] 有声/無声判定 (HuBERT)
    t0 = time.perf_counter()
    vuv0 = model.predict(inputfile)
    timings['[6] HuBERT predict'] = time.perf_counter() - t0

    # [7] DTWアライメント
    t0 = time.perf_counter()
    mapper = {0: 0, 1: 10, 2: 2}
    vuv1 = [mapper[x] for x in vuv0]
    opt = dtw.dtw(vuv1, notes, uv_val=2)
    timings['[7] DTW'] = time.perf_counter() - t0

    # [8] 時間伸縮
    t0 = time.perf_counter()
    orgidx = []
    for i in range(len(note_start)):
        start_pos = opt[note_start[i]][0]
        end_pos = opt[note_end[i]][0]
        idx = stretch.stretch_idx(vuv0[start_pos:(end_pos + 1)], note_end[i] - note_start[i] + 1, start_pos)
        orgidx.extend(idx)
    timings['[8] stretch'] = time.perf_counter() - t0

    # [9] 配列コピー
    t0 = time.perf_counter()
    N = len(pitches)
    sp_out = np.zeros((N * 4, anl["spectrum"].shape[1]))
    ap_out = np.zeros((N * 4, anl["aperiodicity"].shape[1]))
    f0_out = np.zeros(N * 4)
    vu = np.zeros(N * 4)
    org_f0 = np.zeros(N * 4)
    for i in range(N):
        idx = int(orgidx[i])
        src_slice = slice(idx * 4, idx * 4 + 4)
        dst_slice = slice(i * 4, i * 4 + 4)
        sp_out[dst_slice, :] = anl["spectrum"][src_slice, :]
        ap_out[dst_slice, :] = anl["aperiodicity"][src_slice, :]
        f0_out[dst_slice] = pitches[opt[i][1]]
        vu[dst_slice] = vuv0[opt[i][0]]
        org_f0[dst_slice] = anl["f0"][src_slice]
    timings['[9] Array copy'] = time.perf_counter() - t0

    # [10] ビブラート付加
    t0 = time.perf_counter()
    f0_out = np.ascontiguousarray(vibration.add_dumping(f0_out))
    timings['[10] Vibration'] = time.perf_counter() - t0

    # [11] 合成 (PyWORLD)
    t0 = time.perf_counter()
    anl2 = {"f0": f0_out, "spectrum": sp_out, "aperiodicity": ap_out, "rate": sr, "vuv": vu, "org_f0": org_f0}

    # synthesize内の処理を展開
    ap_synth = anl2["aperiodicity"].copy()
    f0_synth = anl2["f0"].copy()
    vuv_synth = anl2["vuv"]
    org_f0_synth = anl2["org_f0"]
    voiced_ap = ap_synth[org_f0_synth > 0].mean(axis=0)
    unvoiced_ap = ap_synth[org_f0_synth == 0].mean(axis=0)

    for i in range(len(vuv_synth)):
        if vuv_synth[i] == 1 and org_f0_synth[i] == 0:
            ap_synth[i, :] = voiced_ap
        elif vuv_synth[i] != 1 and org_f0_synth[i] > 0:
            ap_synth[i, :] = unvoiced_ap
        elif vuv_synth[i] != 1 and org_f0_synth[i] == 0:
            f0_synth[i] = 0
        elif vuv_synth[i] == 0 or f0_synth[i] < 40:
            f0_synth[i] = 0

    w2 = pw.synthesize(f0_synth, anl2["spectrum"], ap_synth, anl2["rate"])
    timings['[11] Synthesize'] = time.perf_counter() - t0

    # ファイル書き出し
    t0 = time.perf_counter()
    sf.write(outputfile, w2, sr)
    timings['[12] File write'] = time.perf_counter() - t0

    total_time = time.perf_counter() - total_start

    return timings, total_time


def print_results(timings, total_time, title="プロファイリング結果"):
    """結果を表示"""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    for name, elapsed in timings.items():
        percent = (elapsed / total_time) * 100
        bar = "#" * int(percent / 2)
        print(f"{name:25s}: {elapsed:7.3f}s ({percent:5.1f}%) {bar}")
    print("-" * 50)
    print(f"{'Total':25s}: {total_time:7.3f}s")
    print("=" * 50)


if __name__ == "__main__":
    inputfile = "test.wav"
    musicxmlfile = "sample.musicxml"
    outputfile = "profile_output.wav"
    modelfile = "hubert_sad_20ms_model.pth"
    modelname = "facebook/hubert-base-ls960"

    print(f"Input: {inputfile}")
    print(f"MusicXML: {musicxmlfile}")
    print(f"Output: {outputfile}")

    # === 最適化後（1回目: モデル読み込みあり） ===
    timings1, total1 = profile_convert_speech2sing(
        inputfile, musicxmlfile, outputfile, modelfile, modelname,
        use_cached_model=True, use_fast_audio=True
    )
    print_results(timings1, total1, "最適化後 (1回目: モデル初期読み込み)")

    # === 最適化後（2回目: キャッシュ使用） ===
    timings2, total2 = profile_convert_speech2sing(
        inputfile, musicxmlfile, "profile_output2.wav", modelfile, modelname,
        use_cached_model=True, use_fast_audio=True
    )
    print_results(timings2, total2, "最適化後 (2回目: モデルキャッシュ使用)")

    # === 比較 ===
    print("\n" + "=" * 50)
    print("効果まとめ")
    print("=" * 50)
    print(f"1回目（初期読み込み）: {total1:.3f}s")
    print(f"2回目（キャッシュ使用）: {total2:.3f}s")
    print(f"2回目の高速化: {(1 - total2/total1)*100:.1f}%")
    print("=" * 50)
