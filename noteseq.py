import pandas as pd
import numpy as np
import librosa


def decay_vectorized(x, alpha=0.75, high=10, low=2):
    """
    Vectorized decay function using NumPy operations.
    Much faster than applying decay0 element by element.
    """
    return np.where(x < alpha, high, high + (low - high) * (x - alpha) / (1 - alpha))


# Keep original functions for backward compatibility
def decay0(x, alpha=0.75, high=10, low=2):
    """
    Calculate decay value for a single value x.
    """
    if x < alpha:
        return high
    else:
        return high + (low - high) * (x - alpha) / (1 - alpha)


def decay(x, alpha=0.75):
    """
    Apply decay0 to each element of array x.
    Now uses vectorized implementation for speed.
    """
    return decay_vectorized(x, alpha, 10, 2)


def noteseq(x, high=10, low=2):
    """
    Generate note state sequence from DataFrame.

    Args:
        x: DataFrame with 'pitch', 'duration', 'start', 'end' columns
        high: high value for decay (default: 10)
        low: low value for decay (default: 2)

    Returns:
        tuple: (notes_array, pitches_array, start_series, end_series)
    """
    # Pre-calculate total size for array allocation
    total_duration = x['duration'].sum()
    out = np.zeros(total_duration)
    p = np.zeros(total_duration)

    # Cache for pitch calculations
    pitch_cache = {}

    pos = 0
    for index, row in x.iterrows():
        pitchsym = row['pitch']
        duration = row['duration']

        if pitchsym == "r":
            # Rest: zeros (already initialized)
            pass
        else:
            # Note: apply decay envelope
            # Use cache to avoid redundant librosa.note_to_hz calls
            if pitchsym not in pitch_cache:
                pitch_cache[pitchsym] = librosa.note_to_hz(pitchsym)
            pitch = pitch_cache[pitchsym]

            seq_vals = np.linspace(0, 1, num=duration)
            out[pos:pos+duration] = decay_vectorized(seq_vals, 0.75, high, low)
            p[pos:pos+duration] = pitch

        pos += duration

    return (out, p, x['start'], x['end'])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create test DataFrame
    notes = pd.DataFrame({
        'pitch': ["c4", "d4", "e4", "r", "c4", "d4", "e4", "r", "g4", "e4", "d4", "c4", "d4", "e4", "d4", "r"],
        'duration': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        'start': list(range(0, 96, 6)),
        'end': list(range(5, 101, 6))
    })

    # Run function
    x, p, start, end = noteseq(notes)
    print(x)
    plt.plot(x)
    plt.show()
