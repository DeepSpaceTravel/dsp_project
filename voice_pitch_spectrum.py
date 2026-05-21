"""
即時麥克風錄音 → 存檔 → 提頻（+4 semitones）→ 頻譜箱型圖
使用方式：
    pip install sounddevice librosa soundfile numpy scipy matplotlib
    python voice_pitch_spectrum.py
"""

import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi
import sys
import os

# ─── 參數設定 ────────────────────────────────────────────────
RECORD_SECONDS = 5          # 錄音秒數（可自行調整）
RATE           = 44100      # 採樣率
CHANNELS       = 1          # 單聲道
CHUNK          = 2048       # FFT 視窗大小

PITCH_SEMITONES = 4         # 提高幾個半音

CUTOFF_FREQ = 80.0          # 高通濾波截止頻率 (Hz)
NUM_BINS    = 24            # 對數頻段數量
FREQ_MIN    = 20.0
FREQ_MAX    = 20000.0

RAW_FILE    = "recording_raw.wav"
SHIFTED_FILE = "recording_shifted.wav"


# ─── 1. 錄音 ─────────────────────────────────────────────────
def record_audio(filename: str, duration: int, rate: int, channels: int) -> np.ndarray:
    print(f"🎙  開始錄音，請說話（{duration} 秒）...")
    audio = sd.rec(
        int(duration * rate),
        samplerate=rate,
        channels=channels,
        dtype="float32",
    )
    sd.wait()
    print(f"✅ 錄音完成，儲存至 {filename}")
    sf.write(filename, audio, rate)
    return audio.squeeze()   # shape: (N,)


# ─── 2. 音高移調 ──────────────────────────────────────────────
def pitch_shift(audio: np.ndarray, rate: int, n_steps: int, filename: str) -> np.ndarray:
    print(f"🎵 正在提頻 +{n_steps} 個半音，請稍候...")
    # librosa.effects.pitch_shift 需要 float32 的 1-D array
    shifted = librosa.effects.pitch_shift(
        audio.astype(np.float32),
        sr=rate,
        n_steps=n_steps,
    )
    sf.write(filename, shifted, rate)
    print(f"✅ 提頻完成，儲存至 {filename}")
    return shifted


# ─── 3. 建立頻段索引（對數等分）──────────────────────────────
def build_log_bins(chunk: int, rate: int, num_bins: int, freq_min: float, freq_max: float):
    x_freq     = np.fft.rfftfreq(chunk, 1.0 / rate)
    log_edges  = np.logspace(np.log10(freq_min), np.log10(freq_max), num_bins + 1)
    bin_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

    band_indices = []
    for i in range(num_bins):
        idx = np.where((x_freq >= log_edges[i]) & (x_freq < log_edges[i + 1]))[0]
        if len(idx) == 0:
            idx = np.array([0])
        band_indices.append(idx)

    xtick_labels = []
    for f in bin_centers:
        if f < 1000:
            xtick_labels.append(f"{int(f)}")
        else:
            xtick_labels.append(f"{f / 1000:.1f}k")

    return band_indices, bin_centers, xtick_labels


# ─── 4. 計算單一音訊的頻段箱型資料 ──────────────────────────
def compute_boxplot_data(
    audio: np.ndarray,
    rate: int,
    chunk: int,
    band_indices: list,
    sos,
) -> list:
    """將整段音訊切成 chunks 並累積每個頻段的 dB 分布。"""
    window   = np.hanning(chunk)
    all_bins = [[] for _ in range(len(band_indices))]
    zi       = sosfilt_zi(sos) * 0.0

    # 逐 chunk 處理（不足一個 chunk 的尾段直接捨棄）
    n_chunks = len(audio) // chunk
    for i in range(n_chunks):
        seg = audio[i * chunk : (i + 1) * chunk]
        filtered, zi = sosfilt(sos, seg, zi=zi)
        windowed     = filtered * window
        fft_result   = np.fft.rfft(windowed)
        magnitude    = np.abs(fft_result) / (chunk / 2)
        db_data      = 20 * np.log10(magnitude + 1e-10)

        for j, idx in enumerate(band_indices):
            all_bins[j].extend(db_data[idx].tolist())

    # 若每個 bin 沒有資料則填 -100（靜音底）
    box_data = [np.array(b) if b else np.array([-100.0]) for b in all_bins]
    return box_data


# ─── 5. 繪製對比頻譜圖 ───────────────────────────────────────
def plot_comparison(
    raw_box:     list,
    shifted_box: list,
    bin_centers: np.ndarray,
    xtick_labels: list,
    num_bins: int,
    cutoff_freq: float,
    pitch_semitones: int,
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("#1e1e1e")
    fig.suptitle(
        f"Human Voice Spectrum — Raw vs Pitch-Shifted (+{pitch_semitones} semitones)",
        color="white", fontsize=14, y=0.98,
    )

    datasets   = [raw_box, shifted_box]
    subtitles  = ["原始錄音 (Raw)", f"提頻後 (+{pitch_semitones} 半音)"]
    positions  = range(1, num_bins + 1)
    display_labels = [
        label if i % 2 == 0 else "" for i, label in enumerate(xtick_labels)
    ]

    for ax, box_data, subtitle in zip(axes, datasets, subtitles):
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.set_title(subtitle, color="white", fontsize=12)
        ax.set_ylabel("Relative Intensity (dBFS)", color="white")
        ax.set_ylim(-100, 0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        box = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels, color="white", rotation=45)

        # 依頻段上色
        for i, patch in enumerate(box["boxes"]):
            freq = bin_centers[i]
            if freq < cutoff_freq:
                color = "#444444"       # 被濾掉的極低頻
            elif 85 <= freq <= 255:
                color = "#ffaa00"       # 基礎音調（F0）
            elif 255 < freq <= 8000:
                color = "#00e5ff"       # 共振峰 & 子音
            else:
                color = "#2a5a5a"       # 極高頻
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for median in box["medians"]:
            median.set(color="white", linewidth=1.5)
        for item in ["whiskers", "caps"]:
            for obj in box[item]:
                obj.set(color="#888888")

    axes[-1].set_xlabel("Frequency (Hz)", color="white")

    plt.tight_layout()
    plt.savefig("spectrum_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="#1e1e1e")
    print("📊 頻譜對比圖已儲存至 spectrum_comparison.png")
    plt.show()


# ─── 主程式 ──────────────────────────────────────────────────
def main():
    # 建立高通濾波器（與原程式相同）
    sos = butter(N=6, Wn=CUTOFF_FREQ, btype="highpass", fs=RATE, output="sos")

    # 建立頻段索引
    band_indices, bin_centers, xtick_labels = build_log_bins(
        CHUNK, RATE, NUM_BINS, FREQ_MIN, FREQ_MAX
    )

    # 步驟 1：錄音
    raw_audio = record_audio(RAW_FILE, RECORD_SECONDS, RATE, CHANNELS)

    # 步驟 2：提頻
    shifted_audio = pitch_shift(raw_audio, RATE, PITCH_SEMITONES, SHIFTED_FILE)

    # 步驟 3：計算頻段資料
    print("🔍 計算頻譜中...")
    raw_box     = compute_boxplot_data(raw_audio,     RATE, CHUNK, band_indices, sos)
    shifted_box = compute_boxplot_data(shifted_audio, RATE, CHUNK, band_indices, sos)

    # 步驟 4：繪圖
    plot_comparison(
        raw_box, shifted_box, bin_centers, xtick_labels,
        NUM_BINS, CUTOFF_FREQ, PITCH_SEMITONES,
    )


if __name__ == "__main__":
    main()
