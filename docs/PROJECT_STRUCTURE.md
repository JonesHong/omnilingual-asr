# 專案文件結構說明

## 📁 目錄結構

```
omnilingual-asr/
├── src/                          # 核心源代碼
│   └── omnilingual_asr/
│       ├── streaming.py          # CTC 串流轉譯（Stride-based）
│       ├── streaming_vad.py      # LLM 串流轉譯（VAD-based）
│       ├── enhanced_pipeline.py  # 增強版 Pipeline（自動切割）
│       └── text_utils.py         # 文字處理工具（去重）
│
├── demos/                        # 示範應用
│   ├── web_streaming_server.py  # Web 串流轉譯伺服器
│   ├── microphone_streaming.py  # 麥克風串流示範
│   └── run_web_demo.sh          # 啟動 Web Demo 腳本
│
├── examples/                     # 使用範例
│   ├── transcribe_long_audio.py # 長音訊轉譯範例
│   └── debug_lang_parameter.py  # 語言參數測試
│
├── tests/                        # 測試文件
│   ├── test_streaming.py        # 串流功能測試
│   ├── test_streaming_impl.py   # 串流實作測試
│   ├── test_streaming_vad.py    # VAD 串流測試
│   ├── test_enhanced_pipeline.py # 增強 Pipeline 測試
│   └── test_robustness.py       # 穩定性測試
│
├── benchmarks/                   # 性能測試
│   └── streaming_performance.py # 串流性能測試
│
├── docs/                         # 文檔
│   ├── ENHANCED_PIPELINE_GUIDE.md      # 增強 Pipeline 指南
│   ├── TYPING_ANIMATION_GUIDE.md       # 打字動畫配置
│   └── CONTRIBUTION_GUIDE_ZH.md        # 貢獻指南（中文）
│
├── self_results/                 # 測試結果與規格文檔
│   ├── streaming_asr_final_spec.md     # 串流 ASR 最終規格
│   ├── streaming_asr_implementation.md # 實作文檔
│   ├── transcription_result*.txt       # 各模型轉譯結果
│   └── ...
│
├── README.md                     # 英文說明
├── README_ZH.md                  # 繁體中文說明
├── requirements_web.txt          # Web Demo 依賴
└── 什麼是上帝的道.mp3            # 測試音訊檔案
```

## 🎯 快速導航

### 想要使用串流轉譯？
→ 查看 `demos/web_streaming_server.py`
→ 執行 `bash demos/run_web_demo.sh`

### 想要轉譯長音訊？
→ 查看 `examples/transcribe_long_audio.py`
→ 使用 `EnhancedASRPipeline`

### 想要了解實作細節？
→ 查看 `self_results/streaming_asr_final_spec.md`
→ 查看 `docs/ENHANCED_PIPELINE_GUIDE.md`

### 想要測試性能？
→ 執行 `python benchmarks/streaming_performance.py`

### 想要貢獻代碼？
→ 閱讀 `docs/CONTRIBUTION_GUIDE_ZH.md`

## 📝 文件命名規則

### 源代碼 (`src/`)
- 使用 snake_case
- 描述性命名
- 例如：`streaming_vad.py`, `enhanced_pipeline.py`

### 示範 (`demos/`)
- 使用 snake_case
- 以用途命名
- 例如：`web_streaming_server.py`, `microphone_streaming.py`

### 範例 (`examples/`)
- 使用 snake_case
- 以功能命名
- 例如：`transcribe_long_audio.py`, `debug_lang_parameter.py`

### 測試 (`tests/`)
- 以 `test_` 開頭
- 描述測試對象
- 例如：`test_streaming.py`, `test_robustness.py`

### 文檔 (`docs/`)
- 使用 UPPER_SNAKE_CASE.md
- 描述性標題
- 例如：`ENHANCED_PIPELINE_GUIDE.md`

## 🔄 遷移指南

### 舊文件名 → 新文件名

| 舊位置 | 新位置 | 說明 |
|--------|--------|------|
| `test_asr_audio.py` | `examples/transcribe_long_audio.py` | 長音訊轉譯範例 |
| `demo_microphone.py` | `demos/microphone_streaming.py` | 麥克風示範 |
| `server.py` | `demos/web_streaming_server.py` | Web 伺服器 |
| `run_demo.sh` | `demos/run_web_demo.sh` | 啟動腳本 |
| `debug_lang.py` | `examples/debug_lang_parameter.py` | 語言參數測試 |
| `tests/benchmark_streaming.py` | `benchmarks/streaming_performance.py` | 性能測試 |

### 更新導入路徑

如果您的代碼引用了舊文件，請更新：

```python
# 舊
from server import ...

# 新
from demos.web_streaming_server import ...
```

### 更新腳本路徑

```bash
# 舊
bash run_demo.sh

# 新
bash demos/run_web_demo.sh
```

## 📚 相關文檔

- [繁體中文使用指南](../README_ZH.md)
- [增強 Pipeline 指南](./ENHANCED_PIPELINE_GUIDE.md)
- [貢獻指南](./CONTRIBUTION_GUIDE_ZH.md)
