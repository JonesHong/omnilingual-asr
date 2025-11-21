# Test Results

此目錄存放測試腳本的執行結果，用於參考和驗證。

## 文件命名規則

- `*_reference.txt` - 黃金標準輸出，提交到 git 作為參考
- `*_latest.txt` - 最新執行結果，提交到 git
- `*_YYYYMMDD_HHMMSS.txt` - 帶時間戳的執行結果，不提交（本地調試用）

## 測試腳本對應

| 測試腳本 | 參考結果文件 | 說明 |
|---------|-------------|------|
| `test_enhanced_pipeline.py` | `test_enhanced_pipeline_reference.txt` | EnhancedASRPipeline 功能測試 |
| `test_streaming.py` | `test_streaming_reference.txt` | StreamingASRPipeline 雙層緩衝測試 |
| `test_robustness.py` | `test_robustness_reference.txt` | 穩健性測試（記憶體洩漏檢測） |

## 使用方式

1. 執行測試腳本會自動保存結果到此目錄
2. 比較新舊結果以驗證功能是否正常
3. 重大更新後，更新 `*_reference.txt` 文件

## 注意事項

- 結果文件可能包含模型輸出，不同運行可能略有差異
- 時間戳和路徑等動態內容會因環境而異
- 主要關注功能邏輯和格式是否正確
