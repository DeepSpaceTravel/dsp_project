import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import queue
import sys
from scipy.signal import butter, sosfilt, sosfilt_zi

# --- 參數設定 ---
CHUNK = 2048
RATE = 44100
CUTOFF_FREQ = 80.0  # 人聲檢測，截斷頻率調降至 80Hz，濾除極低頻與冷氣風切聲

NUM_BINS = 24  # 將 20Hz~20kHz 用對數等分為 24 個箱型圖 (解析度較高)
FREQ_MIN = 20.0
FREQ_MAX = 20000.0

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[:, 0].copy())

# --- 1. 設計軟體高通濾波器 ---
sos = butter(N=6, Wn=CUTOFF_FREQ, btype='highpass', fs=RATE, output='sos')
zi_state = sosfilt_zi(sos) * 0.0 

# --- 2. 建立對數切割的頻段 (Log-spaced Bins) ---
# 產生對數等分的邊界值
log_edges = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_BINS + 1)

# 計算每個箱子的中心頻率 (用於 X 軸標籤顯示)
bin_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

# --- 準備繪圖環境 ---
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

x_freq = np.fft.rfftfreq(CHUNK, 1.0/RATE)

# 找出每個頻段在 FFT 結果中對應的陣列索引
band_indices = []
for i in range(NUM_BINS):
    idx = np.where((x_freq >= log_edges[i]) & (x_freq < log_edges[i+1]))[0]
    # 避免某些極窄頻段內沒有 FFT 點，若無點則塞入 0 避免報錯
    if len(idx) == 0:
        idx = np.array([0]) 
    band_indices.append(idx)

window = np.hanning(CHUNK)

# --- 準備 X 軸的顯示標籤 ---
# 為了避免畫面太擠，我們只標示部分刻度
xtick_labels = []
for f in bin_centers:
    if f < 1000:
        xtick_labels.append(f"{int(f)}")
    else:
        xtick_labels.append(f"{f/1000:.1f}k")

print(f"開始收音... (人聲檢測模式，濾除 {CUTOFF_FREQ}Hz 以下低頻)")

try:
    with sd.InputStream(samplerate=RATE, channels=1, blocksize=CHUNK, callback=audio_callback):
        while True:
            while q.qsize() > 1:
                q.get()
            
            try:
                audio_data = q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 濾波與 FFT
            filtered_data, zi_state = sosfilt(sos, audio_data, zi=zi_state)
            windowed_data = filtered_data * window
            fft_result = np.fft.rfft(windowed_data)
            
            magnitude = np.abs(fft_result) / (CHUNK / 2)
            db_data = 20 * np.log10(magnitude + 1e-10)
            
            # 將數據分裝至各對數頻段
            box_data = [db_data[idx] for idx in band_indices]
            
            # --- 更新圖表 ---
            ax.clear()
            
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            
            ax.set_title('Human Voice Spectrum (Log Scale Boxplots)', color='white', fontsize=14)
            ax.set_ylabel('Relative Intensity (dBFS)', color='white')
            ax.set_xlabel('Frequency (Hz)', color='white')
            ax.set_ylim(-100, 0)
            
            # 繪製箱型圖。
            # 注意：這裡 positions 採用我們計算出來的對數邊界對應的值，讓每個箱子的寬度看起來等距。
            # boxplot 預設是等距排列 (1, 2, 3...)，這正是對數等分 (Log Scale) 視覺上的正確表現法。
            box = ax.boxplot(box_data, positions=range(1, NUM_BINS + 1), widths=0.6, patch_artist=True, showfliers=False)
            
            # 設定 X 軸刻度，每隔 2 個箱子顯示一個標籤，避免重疊
            ax.set_xticks(range(1, NUM_BINS + 1))
            display_labels = [label if i % 2 == 0 else "" for i, label in enumerate(xtick_labels)]
            ax.set_xticklabels(display_labels, color='white', rotation=45)
            
            # 依據人聲頻帶上色
            for i, patch in enumerate(box['boxes']):
                freq = bin_centers[i]
                if freq < CUTOFF_FREQ:
                    # 被濾掉的極低頻
                    color = '#444444' 
                elif 85 <= freq <= 255:
                    # 說話的基礎頻率區間 (Fundemental Frequency, 男女聲)
                    color = '#ffaa00'
                elif 255 < freq <= 8000:
                    # 說話的共振峰與子音區間 (Formants & Consonants)
                    color = '#00e5ff'
                else:
                    # 人聲極少涵蓋的極高頻
                    color = '#2a5a5a'
                    
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
                
            for median in box['medians']:
                median.set(color='white', linewidth=1.5)
            for item in ['whiskers', 'caps']:
                for obj in box[item]:
                    obj.set(color='#888888')

            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\n收到中斷訊號，停止收音。")
finally:
    plt.ioff()
    plt.show()