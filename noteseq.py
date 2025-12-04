import pandas as pd
import numpy as np
import librosa

# decay0 関数
def decay0(x, alpha=0.75, high=10, low=2):
    """
    単一の値xに基づいて減衰値を計算する。
    """
    if x < alpha:
        return high
    else:
        return high + (low - high) * (x - alpha) / (1 - alpha)

# decay 関数 (Rのsapplyに対応)
def decay(x, alpha=0.75):
    """
    配列xの各要素にdecay0を適用する。
    """
    # NumPyのベクトル化を利用して高速に処理
    # (またはリスト内包表記: [decay0(val, alpha) for val in x])
    return np.array([decay0(val, alpha, 10, 2) for val in x])

# noteseq 関数
def noteseq(x, high=10, low=2):
    """
    DataFrame（x）から音符の状態シーケンスを生成する。
    """
    out = []
    p = []
    
    # DataFrameの行を反復処理
    for index, row in x.iterrows():
        pitchsym = row['pitch']
        duration = row['duration']
        
        if pitchsym == "r":
            # rest (休符) はゼロのシーケンス
            out.extend([0] * duration)
            p.extend([0] * duration)
        else:
            # pitch (音符) は減衰シーケンス
            # durationの長さで0から1まで等間隔に分割したシーケンスを生成
            # length.out=duration は duration 個の要素を生成
            pitch = librosa.note_to_hz(pitchsym)
            seq_vals = np.linspace(0, 1, num=duration)
            out.extend(decay(seq_vals))
            p.extend([pitch]*duration)

    return (np.array(out),np.array(p),x['start'],x['end']) # Pythonのリストを最後にNumPy配列に変換して返す

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # データフレームの作成 (Rのdata.frameに対応)
    notes = pd.DataFrame({
        'pitch': ["c4", "d4", "e4", "r", "c4", "d4", "e4", "r", "g4", "e4", "d4", "c4", "d4", "e4", "d4", "r"],
        'duration': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    })

    # 関数の実行
    x,p = noteseq(notes)
    print(x)
    plt.plot(x)
    plt.show()
    