# Web Demo 使用說明

## 啟動方式

```bash
cd demos
python app.py
```

伺服器將在 `http://localhost:8000` 啟動。

## 功能說明

### 1. 檔案轉譯模式 (File Upload)
- **上傳方式**：拖拽或點擊選擇音訊文件
- **支援格式**：所有 torchaudio 支援的格式 (wav, mp3, flac, etc.)
- **基本參數**：
  - Model: 選擇 ASR 模型
  - Language: 選擇目標語言
- **進階參數** (點擊 "⚙️ Advanced" 展開)：
  - **Timestamp Format**: 時間戳格式 (None, Simple [00:05], Detailed [00:05 - 00:08])
  - **Time Format**: 時間顯示格式 (Auto, Seconds, MM:SS, HH:MM:SS)
  - **Chunk Duration**: 分塊長度 (預設 30s)
  - **Overlap**: 分塊重疊 (預設 1s)

### 2. 麥克風轉譯模式 (Microphone)
- **即時串流**：使用 WebSocket 進行低延遲轉譯
- **雙層顯示**：
  - **灰色斜體**：即時預覽 (低延遲，可能不準確)
  - **黑色正常**：穩定結果 (高準確度)
- **基本參數**：
  - Model: 選擇 ASR 模型
  - Language: 選擇目標語言
- **進階參數**：
  - **Context Duration**: 上下文長度 (影響穩定性，預設 4000ms)
- **流程**：
  1. 選擇模型和語言
  2. 點擊 "Start" 按鈕
  3. 允許瀏覽器訪問麥克風
  4. 開始說話，即時看到轉譯結果
  5. 點擊 "Stop" 停止錄音
  6. 點擊 "Clear" 清空結果

## 技術架構

### 後端 (FastAPI)
- **`demos/app.py`**: 主應用程式
  - `ModelManager`: 動態模型管理，避免重複載入
  - `/api/config`: 獲取可用模型和語言列表
  - `/api/load_model`: 動態切換模型
  - `/api/transcribe_file`: 文件轉譯 API
  - `/ws/transcribe_stream`: WebSocket 串流轉譯

### 前端 (HTML/CSS/JS)
- **`demos/static/index.html`**: 主頁面結構
- **`demos/static/css/style.css`**: 現代化樣式
- **`demos/static/js/api.js`**: API 交互封裝
- **`demos/static/js/audio.js`**: 音訊處理和 WebSocket 管理
- **`demos/static/js/app.js`**: 主應用邏輯和 UI 控制

## 優化特性

1. **模型共享**: 多個 WebSocket 連線共享同一個已載入的模型，節省記憶體
2. **動態切換**: 可以在不重啟伺服器的情況下切換模型
3. **響應式設計**: 支援桌面和移動設備
4. **即時反饋**: 麥克風模式下的雙層顯示提供最佳使用體驗

## 注意事項

- 首次載入模型需要時間，請耐心等待
- 麥克風模式需要 HTTPS 或 localhost (瀏覽器安全限制)
- 切換模型時會清空 GPU 記憶體並重新載入
