# Streaming ASR 開發指南

## 📖 簡介

本模組提供兩套核心串流轉譯 Pipeline，分別針對不同的模型架構與應用場景進行優化：

1. **`StreamingASRPipeline`** (基於 Stride)
   - 適用於 **CTC 模型** (如 `omniASR_CTC_1B`)
   - 特點：極低延遲、逐塊輸出、無 VAD 依賴
   - 原理：使用 Stride 機制處理邊界效應

2. **`StreamingASRPipelineVAD`** (基於 VAD)
   - 適用於 **LLM 模型** (如 `omniASR_LLM_3B`)
   - 特點：高準確度、支援語言條件 (Language Conditioning)
   - 原理：使用 VAD 切割語音片段，累積足夠上下文後送入 LLM

## 🚀 快速開始

### 1. 使用 CTC 模型 (StreamingASRPipeline)

適用於需要極致即時性的場景。

```python
import torch
from omnilingual_asr.streaming import StreamingASRPipeline

# 1. 初始化 Pipeline
pipeline = StreamingASRPipeline(
    model_card="omniASR_CTC_1B",
    device="cuda",
    dtype=torch.float16,
    chunk_size=0.32,  # 每次處理 320ms
    stride=0.08       # 保留 80ms 作為上下文
)

# 2. 模擬串流輸入 (假設 audio_generator() 產生音訊塊)
for chunk in audio_generator():
    # chunk: torch.Tensor, shape (N,)
    
    # 3. 處理音訊塊
    segment = pipeline.process_audio(chunk)
    
    # 4. 獲取結果 (可能為空字串)
    if segment:
        print(segment, end="", flush=True)
```

### 2. 使用 LLM 模型 (StreamingASRPipelineVAD)

適用於需要高準確度或特定語言優化的場景。

```python
import torch
from omnilingual_asr.streaming_vad import StreamingASRPipelineVAD

# 1. 初始化 Pipeline
pipeline = StreamingASRPipelineVAD(
    model_card="omniASR_LLM_3B",
    device="cuda",
    lang="cmn_Hant",  # 指定語言 (繁體中文)
    max_segment_duration_ms=1500  # 強制斷句長度
)

# 2. 模擬串流輸入
for chunk in audio_generator():
    # 3. 處理音訊塊
    # VAD Pipeline 會內部緩衝，直到偵測到完整語句
    text = pipeline.process_audio(chunk)
    
    # 4. 獲取結果 (僅在語句結束或強制斷句時返回)
    if text:
        print(text, end=" ", flush=True)
```

## ⚙️ API 詳解

### 1. StreamingASRPipeline (CTC)

位於 `src/omnilingual_asr/streaming.py`。

#### 初始化參數

```python
def __init__(
    self,
    model_card: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    chunk_size: float = 0.32,  # 秒
    stride: float = 0.08       # 秒
):
```

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `chunk_size` | 每次推論的音訊長度 | `0.32` (320ms) |
| `stride` | 用於處理邊界的重疊長度 | `0.08` (80ms) |

> **注意**：`chunk_size` 越小延遲越低，但計算開銷越大。`stride` 必須足夠大以覆蓋模型的感受野 (Receptive Field)。

#### 方法

- **`process_audio(waveform: torch.Tensor) -> str`**
  - 輸入：單聲道音訊張量 (16kHz)
  - 輸出：該片段解碼出的文字 (若無新文字則返回空字串)

---

### 2. StreamingASRPipelineVAD (LLM)

位於 `src/omnilingual_asr/streaming_vad.py`。

#### 初始化參數

```python
def __init__(
    self,
    model_card: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    lang: str = "eng_Latn",
    # VAD 參數
    sample_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    max_segment_duration_ms: int = 2000
):
```

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `lang` | 目標語言代碼 | `cmn_Hant` (繁中), `eng_Latn` (英) |
| `max_segment_duration_ms` | 強制斷句的最大長度 | `1500` - `2000` |
| `min_silence_duration_ms` | 判斷語句結束的靜音長度 | `300` - `500` |

#### 方法

- **`process_audio(chunk: torch.Tensor) -> str`**
  - 輸入：任意長度的音訊塊 (建議 30ms - 100ms)
  - 輸出：完整語句的轉譯結果 (僅在 VAD 觸發時返回)
  - **行為**：此方法會將音訊寫入內部緩衝區，並更新 VAD 狀態。只有當 VAD 偵測到語句結束 (End of Speech) 或緩衝區超過 `max_segment_duration_ms` 時，才會觸發 LLM 推論並返回文字。

## 🔧 進階實作細節

### Stride 機制 (CTC)

CTC 模型是逐幀輸出的，但直接切割音訊會導致邊界處的文字被切斷或識別錯誤。`StreamingASRPipeline` 使用 **Right Context Stride** 技術：

1. 每次輸入 `chunk_size` 的音訊。
2. 模型推論後，丟棄最後 `stride` 秒的輸出 (因為缺乏未來上下文，不準確)。
3. 保留剩餘部分的輸出。
4. 下一次輸入時，從上一次丟棄的位置開始讀取。

### VAD 機制 (LLM)

LLM 模型是 Sequence-to-Sequence 的，需要完整的上下文才能準確輸出。因此不能像 CTC 那樣逐幀處理。`StreamingASRPipelineVAD` 使用 **Silero VAD**：

1. 持續緩衝輸入音訊。
2. VAD 實時偵測語音活動。
3. 當偵測到 **"語音結束" (Silence)** 時，將緩衝的語音片段送入 LLM。
4. 為了降低延遲，若語音持續太久 (超過 `max_segment_duration_ms`)，會強制切割並送出。

## 📊 性能比較

| 特性 | CTC Pipeline | VAD Pipeline (LLM) |
|------|--------------|-------------------|
| **模型** | `omniASR_CTC_*` | `omniASR_LLM_*` |
| **延遲** | 極低 (< 300ms) | 中等 (1-2秒) |
| **準確度** | 中等 | 高 |
| **語言支援** | 自動識別 (不可指定) | **需指定語言 (`lang`)** |
| **輸出方式** | 逐字流式輸出 | 整句輸出 |

## 💡 開發建議

1. **選擇正確的 Pipeline**：
   - 做即時字幕、語音指令 → 用 **CTC**。
   - 做會議記錄、多語種翻譯 → 用 **LLM**。

2. **處理 LLM 重複問題**：
   - LLM 模型偶爾會產生重複文字。建議在應用層整合 `src/omnilingual_asr/text_utils.py` 中的 `clean_asr_output` 函數進行後處理。

3. **VAD 參數調優**：
   - 如果覺得反應太慢，調低 `min_silence_duration_ms`。
   - 如果覺得句子被切斷，調高 `max_segment_duration_ms`。
