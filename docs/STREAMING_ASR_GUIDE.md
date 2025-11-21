# Streaming ASR 開發指南

## 📖 簡介

本模組提供了一套統一且強大的串流轉譯 Pipeline：**`StreamingASRPipeline`**。

它整合了 **VAD (語音活動偵測)** 與 **雙層緩衝機制 (Two-Pass Strategy)**，旨在解決傳統串流轉譯中常見的「語意截斷」與「高延遲」問題。

### 核心特性

1.  **統一架構**：單一 Pipeline 同時支援 CTC 與 LLM 模型。
2.  **VAD 智慧切割**：利用語音活動偵測，確保在語句完整處進行切割，大幅降低錯誤率。
3.  **雙層緩衝 (Two-Pass Strategy)**：
    *   **Preview (快速路徑)**：低延遲輸出初步結果，讓使用者即時看到文字。
    *   **Stable (慢速路徑)**：累積足夠上下文後進行二次修正，確保最終結果準確無誤。
4.  **靈活配置**：可根據需求開啟或關閉 VAD 與雙層緩衝。

---

## 🚀 快速開始

### 1. 基本使用 (推薦配置)

適用於大多數場景，開啟 VAD 與雙層緩衝以獲得最佳體驗。

```python
import torch
from omnilingual_asr.streaming import StreamingASRPipeline

# 1. 初始化 Pipeline
pipeline = StreamingASRPipeline(
    model_card="omniASR_LLM_3B",  # 或 omniASR_CTC_1B
    device="cuda",
    lang="cmn_Hant",              # 指定語言
    use_vad=True,                 # 開啟 VAD (推薦)
    enable_stability_pass=True,   # 開啟雙層緩衝 (推薦)
    context_duration_ms=10000     # 穩定層上下文長度 (10秒)
)

# 2. 模擬串流輸入
print("開始串流...")
for chunk in audio_generator():
    # 添加音訊到緩衝區
    pipeline.add_audio(chunk)
    
    # 獲取可用結果
    for result in pipeline.transcribe_available():
        if result.is_stable:
            # 穩定結果 (綠色顯示)
            print(f"\033[92m[STABLE] {result.text}\033[0m")
        else:
            # 預覽結果 (灰色顯示)
            print(f"[PREVIEW] {result.text}")

# 3. 結束處理
for result in pipeline.finish():
    print(f"[FINAL] {result.text}")
```

### 2. 低延遲模式 (CTC 模型)

適用於需要極致反應速度的場景（如語音指令），可關閉雙層緩衝。

```python
pipeline = StreamingASRPipeline(
    model_card="omniASR_CTC_300M",
    use_vad=True,
    enable_stability_pass=False,  # 關閉雙層緩衝，直接輸出
    min_speech_duration_ms=100,   # 更靈敏的偵測
    min_silence_duration_ms=300
)
```

---

## ⚙️ API 詳解

### `StreamingASRPipeline`

位於 `src/omnilingual_asr/streaming.py`。

#### 初始化參數

```python
def __init__(
    self,
    model_card: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    lang: Optional[str] = None,
    
    # VAD 參數
    use_vad: bool = True,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    max_segment_duration_ms: int = 2000,
    
    # 雙層緩衝參數
    enable_stability_pass: bool = False,
    context_duration_ms: int = 15000
):
```

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `model_card` | 模型名稱 | `omniASR_LLM_3B` (高準確), `omniASR_CTC_1B` (快速) |
| `lang` | 目標語言代碼 | `cmn_Hant` (繁中), `eng_Latn` (英) |
| `use_vad` | 是否啟用 VAD 切割 | `True` (強烈建議) |
| `enable_stability_pass` | 是否啟用雙層緩衝 | `True` (LLM 模型建議開啟) |
| `context_duration_ms` | 穩定層的上下文長度 | `10000` - `15000` (10-15秒) |

#### 核心方法

- **`add_audio(audio: np.ndarray)`**
  - 輸入：單聲道音訊數據 (numpy array, 16kHz)
  - 功能：將音訊加入內部緩衝區。

- **`transcribe_available() -> Iterator[StreamingResult]`**
  - 功能：處理緩衝區中的音訊，返回可用的轉譯結果。
  - 返回：`StreamingResult` 物件迭代器。

- **`finish() -> Iterator[StreamingResult]`**
  - 功能：處理剩餘的所有音訊，清空緩衝區。

#### `StreamingResult` 物件

```python
@dataclass
class StreamingResult:
    text: str          # 轉譯文字
    is_final: bool     # 是否為該片段的最終結果
    timestamp: float   # 結束時間戳記
    latency: float     # 處理延遲
    start_timestamp: float # 開始時間戳記
    is_stable: bool    # True=穩定結果, False=預覽結果
```

---

## 🔧 架構原理

### 1. VAD 智慧切割

傳統的 Stride 機制每隔固定時間（如 3秒）強制切割，容易切斷單字，導致模型（尤其是 LLM）產生幻覺或拼寫錯誤。

本 Pipeline 使用 **Silero VAD** 實時偵測語音活動，只在 **"語音結束" (Silence)** 或 **"換氣"** 時進行切割。這保證了送入模型的永遠是完整的語句，大幅提升準確率。

### 2. 雙層緩衝 (Two-Pass Strategy)

為了同時滿足「低延遲」與「高準確度」的需求，我們設計了雙層機制：

1.  **第一層 (Preview Pass)**：
    *   當 VAD 偵測到短暫停頓（如 0.5秒）時，立即送出當前片段進行轉譯。
    *   **特點**：快，但因為上下文較短，可能不夠準確。
    *   **用途**：讓使用者感覺系統在即時反應。

2.  **第二層 (Stability Pass)**：
    *   系統會在背景將這些短片段累積起來。
    *   當累積長度達到 `context_duration_ms` (如 10秒) 時，將整段長音訊再次送入模型。
    *   **特點**：慢一點，但因為有長上下文，模型能自我修正之前的錯誤。
    *   **用途**：覆蓋之前的預覽結果，提供最終確定的文本。

---

## 📊 性能比較

| 模式 | 延遲 | 準確度 | 適用場景 |
|------|------|--------|----------|
| **CTC (無 VAD)** | 極低 (< 300ms) | 低 (易切斷) | 簡單指令、關鍵字偵測 |
| **VAD Only** | 中等 (1-2秒) | 高 | 一般對話、短句翻譯 |
| **VAD + Two-Pass** | **動態** (預覽快/修正準) | **極高** | 會議記錄、長篇演講、專業字幕 |

## 💡 開發建議

1.  **LLM 模型強烈建議開啟 `enable_stability_pass`**：LLM 對上下文非常敏感，長上下文能顯著減少幻覺。
2.  **VAD 參數調優**：
    *   如果覺得反應太慢，調低 `min_silence_duration_ms` (例如 300ms)。
    *   如果覺得句子被切斷，調高 `max_segment_duration_ms` (例如 3000ms)。
3.  **前端顯示**：
    *   建議在 UI 上區分「預覽文字」（灰色/斜體）和「穩定文字」（黑色/正常）。
    *   當收到 `is_stable=True` 的結果時，應該替換掉之前對應時間段的預覽文字。
