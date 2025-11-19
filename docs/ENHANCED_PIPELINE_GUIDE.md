# EnhancedASRPipeline 使用指南

## 📖 簡介

`EnhancedASRPipeline` 是 `ASRInferencePipeline` 的增強包裝層，提供：

1. ✅ **自動音訊切割** - 處理任意長度音訊（超過 40 秒自動切割）
2. ✅ **靈活時間戳記** - 支援多種時間戳記格式
3. ✅ **可配置時間格式** - 秒數、MM:SS、HH:MM:SS
4. ✅ **自訂模板** - 完全自訂時間戳記顯示
5. ✅ **不修改核心** - 不影響原始 `ASRInferencePipeline`

## 🚀 快速開始

### 基本使用

```python
from omnilingual_asr.enhanced_pipeline import EnhancedASRPipeline
import torch

# 初始化
pipeline = EnhancedASRPipeline(
    model_card="omniASR_LLM_3B",
    device="cuda",
    dtype=torch.float16
)

# 準備輸入
inp = [{
    "waveform": waveform,  # torch.Tensor
    "sample_rate": 16000
}]

# 轉譯（自動處理長音訊）
result = pipeline.transcribe(
    inp=inp,
    batch_size=1,
    lang=["cmn_Hant"]
)
```

## 📊 時間戳記格式

### 1. 無時間戳記（預設）

```python
result = pipeline.transcribe(
    inp=inp,
    timestamp_format="none"  # 或 TimestampFormat.NONE
)
```

**輸出**：
```
什麼是上帝的道那你應該知道就是上帝的道你沒有說我在說與上帝同在...
```

### 2. 簡易時間戳記

```python
from omnilingual_asr.enhanced_pipeline import TimestampFormat

result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.SIMPLE
)
```

**輸出**：
```
[00:00] 什麼是上帝的道那你應該知道就是上帝的道
[00:30] 你沒有說我在說與上帝同在倒是聖靈待到人家
[01:00] 這個人水聖靈借著墓室就業的先知跟新約的師徒
```

### 3. 詳細時間戳記

```python
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.DETAILED
)
```

**輸出**：
```
[00:00 - 00:30] 什麼是上帝的道那你應該知道就是上帝的道
[00:30 - 01:00] 你沒有說我在說與上帝同在倒是聖靈待到人家
[01:00 - 01:30] 這個人水聖靈借著墓室就業的先知跟新約的師徒
```

## ⏰ 時間格式

### 支援的格式

```python
from omnilingual_asr.enhanced_pipeline import TimeFormat

# 1. 秒數
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.SIMPLE,
    time_format=TimeFormat.SECONDS
)
# 輸出: [5.2s] 文字內容

# 2. MM:SS（預設）
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.SIMPLE,
    time_format=TimeFormat.MMSS
)
# 輸出: [00:05] 文字內容

# 3. HH:MM:SS
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.SIMPLE,
    time_format=TimeFormat.HHMMSS
)
# 輸出: [00:00:05] 文字內容
```

## 🎨 自訂模板

### 使用模板變數

可用變數：
- `{start}` - 開始時間
- `{end}` - 結束時間
- `{text}` - 文字內容
- `{duration}` - 持續時間

### 範例

```python
# 範例 1: 簡潔格式
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.SIMPLE,  # 必須非 NONE
    time_format=TimeFormat.SECONDS,
    timestamp_template="{start} | {text}"
)
# 輸出: 5.2s | 文字內容

# 範例 2: 詳細格式
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.MMSS,
    timestamp_template="⏱️ {start} → {end} ({duration})\n{text}"
)
# 輸出:
# ⏱️ 00:00 → 00:30 (00:30)
# 文字內容

# 範例 3: SRT 字幕格式
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.HHMMSS,
    timestamp_template="{start} --> {end}\n{text}"
)
# 輸出:
# 00:00:00 --> 00:00:30
# 文字內容
```

## ⚙️ 音訊切割參數

### 基本參數

```python
result = pipeline.transcribe(
    inp=inp,
    chunk_duration=30.0,  # 每段 30 秒（必須 <= 40）
    overlap=1.0,  # 重疊 1 秒（避免句子被切斷）
)
```

### 參數說明

| 參數 | 說明 | 預設值 | 限制 |
|------|------|--------|------|
| `chunk_duration` | 每段音訊長度（秒） | 30.0 | **必須 <= 40** |
| `overlap` | 重疊長度（秒） | 1.0 | >= 0 |

### 錯誤處理

```python
try:
    result = pipeline.transcribe(
        inp=inp,
        chunk_duration=50.0  # ❌ 超過 40 秒限制
    )
except ValueError as e:
    print(f"錯誤: {e}")
    # 輸出: chunk_duration (50.0s) 不能超過模型限制 (40.0s)
```

## 📝 完整範例

### 處理長音訊檔案

```python
import torch
import torchaudio
from omnilingual_asr.enhanced_pipeline import (
    EnhancedASRPipeline,
    TimestampFormat,
    TimeFormat
)

# 載入音訊
waveform, sr = torchaudio.load("long_audio.mp3")
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# 初始化 Pipeline
pipeline = EnhancedASRPipeline(
    model_card="omniASR_LLM_3B",
    device="cuda",
    dtype=torch.float16
)

# 轉譯（自動切割）
result = pipeline.transcribe(
    inp=[{
        "waveform": waveform.squeeze(0),
        "sample_rate": sr
    }],
    batch_size=1,
    lang=["cmn_Hant"],
    chunk_duration=30.0,
    overlap=1.0,
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.MMSS
)

print(result[0])
```

### 輸出範例

```
[00:00 - 00:30] 什麼是上帝的道那你應該知道就是上帝的道
[00:30 - 01:00] 你沒有說我在說與上帝同在倒是聖靈待到人家
[01:00 - 01:30] 這個人水聖靈借著墓室就業的先知跟新約的師徒
[01:30 - 02:00] 寫下這本書新舊樂生經這個是文字的當叫做真理
```

## 🔧 進階用法

### 批次處理多個檔案

```python
# 準備多個音訊
inp = [
    {"waveform": waveform1, "sample_rate": 16000},
    {"waveform": waveform2, "sample_rate": 16000},
]

# 批次轉譯
results = pipeline.transcribe(
    inp=inp,
    batch_size=1,  # 逐個處理（避免記憶體問題）
    lang=["cmn_Hant", "cmn_Hant"],
    timestamp_format=TimestampFormat.SIMPLE
)

for i, result in enumerate(results):
    print(f"檔案 {i+1}:")
    print(result)
    print()
```

### 生成字幕檔案

```python
# 使用 SRT 格式模板
result = pipeline.transcribe(
    inp=inp,
    timestamp_format=TimestampFormat.DETAILED,
    time_format=TimeFormat.HHMMSS,
    timestamp_template="{start} --> {end}\n{text}"
)

# 加入序號生成完整 SRT
lines = result[0].split('\n\n')
srt_content = []
for i, line in enumerate(lines, 1):
    srt_content.append(f"{i}\n{line}\n")

with open("output.srt", "w", encoding="utf-8") as f:
    f.write("\n".join(srt_content))
```

## ⚡ 性能考量

### 記憶體使用

- 音訊切割是**逐段處理**，不會增加峰值記憶體
- 每次只載入一個 chunk 到 GPU

### 速度優化

```python
# 較大的 chunk_duration 可以減少切割次數
result = pipeline.transcribe(
    inp=inp,
    chunk_duration=35.0,  # 接近上限，減少切割
    overlap=0.5  # 較小的重疊
)
```

### 權衡

| chunk_duration | 優點 | 缺點 |
|----------------|------|------|
| 20-25s | 更精確的時間戳記 | 處理次數多，稍慢 |
| 30-35s | **平衡** | - |
| 35-40s | 最快 | 時間戳記粒度粗 |

## 🐛 常見問題

### Q: 為什麼限制 40 秒？
A: 這是模型的硬性限制，超過會導致錯誤或品質下降。

### Q: overlap 設多少合適？
A: 建議 0.5-2.0 秒。太小可能切斷句子，太大浪費計算。

### Q: 可以關閉自動切割嗎？
A: 不行，但如果音訊 <= 40 秒，不會觸發切割。

### Q: 時間戳記不準確？
A: 檢查 `overlap` 設定。重疊部分的文字會被丟棄，可能影響時間對齊。

## 📚 API 參考

### EnhancedASRPipeline

```python
class EnhancedASRPipeline:
    def __init__(
        self,
        model_card: str = "omniASR_CTC_1B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    )
    
    def transcribe(
        self,
        inp: List[Dict[str, Any]],
        batch_size: int = 1,
        lang: Optional[List[str]] = None,
        chunk_duration: float = 30.0,
        overlap: float = 1.0,
        timestamp_format: TimestampFormat | str = TimestampFormat.NONE,
        time_format: TimeFormat | str = TimeFormat.MMSS,
        timestamp_template: Optional[str] = None
    ) -> List[str]
```

### 枚舉類型

```python
class TimestampFormat(Enum):
    NONE = "none"
    SIMPLE = "simple"
    DETAILED = "detailed"

class TimeFormat(Enum):
    SECONDS = "seconds"
    MMSS = "mm:ss"
    HHMMSS = "hh:mm:ss"
```

## 🎯 總結

`EnhancedASRPipeline` 提供了：

✅ 自動處理長音訊（無需手動切割）
✅ 靈活的時間戳記配置
✅ 不修改核心類（安全）
✅ 簡單易用的 API

適合用於：
- 長音訊轉譯
- 字幕生成
- 會議記錄
- 任何需要時間戳記的場景
