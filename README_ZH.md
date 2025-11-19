# Omnilingual ASR 繁體中文使用指南

> 基於 Meta 的 Omnilingual ASR，添加串流轉譯與自動音訊切割功能

## 🎯 重要更新

本增強版本新增兩大核心功能：

### 1. 🎙️ 即時串流轉譯
- 支援麥克風即時語音辨識
- 低延遲（< 2 秒）
- 支援 CTC 與 LLM 模型
- Web 介面操作簡單

### 2. ✂️ 自動音訊切割
- **不再限制 40 秒**！
- 自動處理任意長度音訊
- 支援可配置時間戳記
- 智能重疊避免句子被切斷

---

## 📦 安裝

### 系統需求

- Python 3.8+
- CUDA 11.8+ （GPU 加速，可選）
- 16GB RAM（建議）

### 安裝步驟

```bash
# 1. 克隆專案
git clone https://github.com/JonesHong/omnilingual-asr.git
cd omnilingual-asr

# 2. 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. 安裝依賴
pip install -e .

# 4. 安裝 Web Demo 依賴（可選）
pip install -r requirements_web.txt
```

---

## 🚀 快速開始

### 功能 1：即時串流轉譯

#### 啟動 Web Demo

```bash
# 啟動伺服器
python demos/web_streaming_server.py
```

然後在瀏覽器開啟：`http://localhost:8000`

#### 配置選項

編輯 `demos/web_streaming_server.py` 調整參數：

```python
# 模型選擇
MODEL_CARD = "omniASR_LLM_3B"  # 或 CTC_300M, LLM_1B 等

# 語言設定
LANG = "cmn_Hant"  # 繁體中文

# VAD 參數（控制延遲）
MAX_SEGMENT_DURATION_MS = 2000  # 最大片段長度
MIN_SILENCE_DURATION_MS = 500   # 靜音等待時間
MIN_SPEECH_DURATION_MS = 250    # 最小語音長度
```

#### 支援的模型

| 模型 | 用途 | 延遲 | 準確度 |
|------|------|------|--------|
| `omniASR_CTC_300M` | 快速辨識 | 極低 | 中 |
| `omniASR_CTC_1B` | 平衡 | 低 | 中高 |
| `omniASR_LLM_300M` | 高準確度 | 中 | 高 |
| `omniASR_LLM_3B` | 最高準確度 | 中高 | 極高 |

#### 語言代碼

常用語言代碼：

- `cmn_Hant` - 繁體中文
- `cmn_Hans` - 簡體中文
- `eng_Latn` - 英文
- `jpn_Jpan` - 日文
- `kor_Hang` - 韓文

完整列表：[lang_ids.py](../src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)

---

### 功能 2：自動音訊切割

#### 基本使用

```python
from omnilingual_asr.enhanced_pipeline import (
    EnhancedASRPipeline,
    TimestampFormat,
    TimeFormat
)
import torchaudio

# 1. 載入音訊（任意長度）
waveform, sr = torchaudio.load("long_audio.mp3")

# 2. 初始化 Pipeline
pipeline = EnhancedASRPipeline(
    model_card="omniASR_LLM_3B",
    device="cuda"  # 或 "cpu"
)

# 3. 轉譯（自動切割）
result = pipeline.transcribe(
    inp=[{
        "waveform": waveform.squeeze(0),
        "sample_rate": sr
    }],
    lang=["cmn_Hant"],
    chunk_duration=30.0,  # 每段 30 秒（必須 <= 40）
    overlap=1.0,  # 重疊 1 秒
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.MMSS
)

print(result[0])
```

#### 輸出範例

**無時間戳記**：
```
什麼是上帝的道那你應該知道就是上帝的道你沒有說我在說與上帝同在...
```

**簡易時間戳記**：
```
[00:00] 什麼是上帝的道那你應該知道就是上帝的道
[00:30] 你沒有說我在說與上帝同在倒是聖靈待到人家
[01:00] 這個人水聖靈借著墓室就業的先知跟新約的師徒
```

**詳細時間戳記**：
```
[00:00 - 00:30] 什麼是上帝的道那你應該知道就是上帝的道
[00:30 - 01:00] 你沒有說我在說與上帝同在倒是聖靈待到人家
[01:00 - 01:30] 這個人水聖靈借著墓室就業的先知跟新約的師徒
```

#### 時間戳記選項

```python
# 1. 無時間戳記
timestamp_format=TimestampFormat.NONE

# 2. 簡易時間戳記
timestamp_format=TimestampFormat.SIMPLE

# 3. 詳細時間戳記
timestamp_format=TimestampFormat.DETAILED

# 時間格式
time_format=TimeFormat.SECONDS   # 5.2s
time_format=TimeFormat.MMSS      # 00:05
time_format=TimeFormat.HHMMSS    # 00:00:05
```

#### 自訂時間戳記模板

```python
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.SIMPLE,
    time_format=TimeFormat.SECONDS,
    timestamp_template="⏱️ {start} | {text}"
)
# 輸出: ⏱️ 5.2s | 文字內容
```

可用變數：
- `{start}` - 開始時間
- `{end}` - 結束時間
- `{text}` - 文字內容
- `{duration}` - 持續時間

---

## 📚 詳細文檔

### 串流轉譯

- [串流轉譯完整指南](./docs/STREAMING_ASR_GUIDE.md)

### 音訊切割

- [EnhancedASRPipeline 完整指南](./docs/ENHANCED_PIPELINE_GUIDE.md)
- [API 參考](../src/omnilingual_asr/enhanced_pipeline.py)

---

## 🎨 進階功能

### 1. 批次處理多個檔案

```python
# 準備多個音訊
files = ["audio1.mp3", "audio2.wav", "audio3.flac"]
inp = []

for file in files:
    waveform, sr = torchaudio.load(file)
    inp.append({
        "waveform": waveform.squeeze(0),
        "sample_rate": sr
    })

# 批次轉譯
results = pipeline.transcribe(
    inp=inp,
    lang=["cmn_Hant"] * len(files),
    timestamp_format=TimestampFormat.SIMPLE
)

for i, result in enumerate(results):
    print(f"\n檔案 {i+1}: {files[i]}")
    print(result)
```

### 2. 生成 SRT 字幕

```python
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.HHMMSS,
    timestamp_template="{start} --> {end}\n{text}"
)

# 加入序號
lines = result[0].split('\n\n')
srt_content = []
for i, line in enumerate(lines, 1):
    srt_content.append(f"{i}\n{line}\n")

with open("output.srt", "w", encoding="utf-8") as f:
    f.write("\n".join(srt_content))
```

### 3. 調整 Web Demo 打字速度

編輯 `./demos/web_streaming_server.py` 第 140 行：

```javascript
const TYPING_SPEED = 30; // 毫秒/字符
```

速度建議：
- `10-20ms` - 極快
- `30-50ms` - 推薦
- `60-80ms` - 慢速

---

## ⚙️ 性能優化

### GPU 記憶體不足

```python
# 使用較小的模型
pipeline = EnhancedASRPipeline(
    model_card="omniASR_CTC_300M",  # 只需 ~2GB
    device="cuda"
)

# 或使用 CPU
pipeline = EnhancedASRPipeline(
    model_card="omniASR_LLM_1B",
    device="cpu"
)
```

### 加快處理速度

```python
# 使用較大的 chunk_duration
result = pipeline.transcribe(
    inp=inp,
    chunk_duration=35.0,  # 接近上限
    overlap=0.5  # 較小的重疊
)
```

### 降低延遲（串流）

```python
# server.py 配置
MAX_SEGMENT_DURATION_MS = 1500  # 降低到 1.5 秒
MIN_SILENCE_DURATION_MS = 300   # 降低到 300ms
```

---

## 🐛 常見問題

### Q: 為什麼串流會有疊字？
A: 已修正！使用了文字去重功能（`text_utils.py`）。如果仍有問題，請檢查 `lang` 參數是否正確（例如 `cmn_Hant` 而非 `cmn`）。

### Q: 音訊切割後時間戳記不準確？
A: 調整 `overlap` 參數。建議 0.5-2.0 秒。

### Q: Web Demo 無法使用麥克風（WSL）？
A: WSL 無法直接訪問 Windows 麥克風。請在 Windows 瀏覽器中開啟 `http://localhost:8000`。

### Q: chunk_duration 可以超過 40 秒嗎？
A: 不行，這是模型的硬性限制。超過會報錯。

### Q: 支援哪些音訊格式？
A: 支援所有 `torchaudio` 支援的格式：MP3, WAV, FLAC, OGG 等。

---

## 📊 模型選擇建議

### 串流轉譯

| 場景 | 推薦模型 | 原因 |
|------|---------|------|
| 即時字幕 | `omniASR_CTC_300M` | 極低延遲 |
| 會議記錄 | `omniASR_LLM_1B` | 平衡準確度與速度 |
| 高品質轉譯 | `omniASR_LLM_3B` | 最高準確度 |

### 檔案轉譯

| 場景 | 推薦模型 | 原因 |
|------|---------|------|
| 快速草稿 | `omniASR_CTC_1B` | 快速 |
| 正式文檔 | `omniASR_LLM_3B` | 高準確度 |
| 多語言混合 | `omniASR_LLM_7B` | 最佳語言識別 |

---

## 🎯 使用範例

### 範例 1：會議記錄

```python
from omnilingual_asr.enhanced_pipeline import *
import torchaudio

# 載入會議錄音（可能很長）
waveform, sr = torchaudio.load("meeting.mp3")

pipeline = EnhancedASRPipeline(
    model_card="omniASR_LLM_3B",
    device="cuda"
)

result = pipeline.transcribe(
    inp=[{"waveform": waveform.squeeze(0), "sample_rate": sr}],
    lang=["cmn_Hant"],
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.MMSS
)

# 儲存結果
with open("meeting_transcript.txt", "w", encoding="utf-8") as f:
    f.write(result[0])
```

### 範例 2：即時字幕

```bash
# 啟動 Web Demo
python server.py

# 在 server.py 中配置
MODEL_CARD = "omniASR_CTC_1B"  # 低延遲
LANG = "cmn_Hant"
MAX_SEGMENT_DURATION_MS = 1500  # 1.5 秒
```

### 範例 3：影片字幕生成

```python
# 從影片提取音訊（需要 ffmpeg）
import subprocess
subprocess.run([
    "ffmpeg", "-i", "video.mp4",
    "-vn", "-acodec", "pcm_s16le",
    "-ar", "16000", "-ac", "1",
    "audio.wav"
])

# 轉譯並生成 SRT
waveform, sr = torchaudio.load("audio.wav")
result = pipeline.transcribe(
    inp=[{"waveform": waveform.squeeze(0), "sample_rate": sr}],
    lang=["cmn_Hant"],
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.HHMMSS,
    timestamp_template="{start} --> {end}\n{text}"
)

# 儲存 SRT
lines = result[0].split('\n\n')
with open("subtitles.srt", "w", encoding="utf-8") as f:
    for i, line in enumerate(lines, 1):
        f.write(f"{i}\n{line}\n\n")
```

---

## 📝 授權

本專案基於 Meta 的 Omnilingual ASR，遵循 [Apache 2.0 授權](./LICENSE)。

增強功能（串流轉譯、自動切割）由社群貢獻，同樣採用 Apache 2.0 授權。

---

## 🙏 致謝

- **Meta AI** - 原始 Omnilingual ASR 模型
- **Fairseq2** - 模型框架
- **社群貢獻者** - 串流與切割功能

---

## 📮 支援

- **問題回報**：[GitHub Issues](https://github.com/omnilingual/omnilingual-asr/issues)
- **功能建議**：[GitHub Discussions](https://github.com/omnilingual/omnilingual-asr/discussions)
- **文檔**：[docs/](./README.md)

---

## 🗺️ 路線圖

- [x] 串流轉譯（CTC + LLM）
- [x] 自動音訊切割
- [x] 繁體中文文檔

---

**享受使用 Omnilingual ASR Enhanced！** 🎉
