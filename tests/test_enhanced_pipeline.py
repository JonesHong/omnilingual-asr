#!/usr/bin/env python3
"""
測試 EnhancedASRPipeline 的音訊切割與時間戳記功能

執行結果會自動保存在 tests/results/ 目錄下。

預期行為：
- 測試 1: 無時間戳記，純文字輸出
- 測試 2-3: 固定時間格式 (MM:SS, HH:MM:SS)
- 測試 4: 自訂時間戳記模板
- 測試 5: JSON 格式輸出，包含 VAD 切割驗證
- 測試 6: 錯誤處理，chunk_duration 超過限制應拋出異常
- 測試 7: 自動時間格式 (長音訊 >60s) 應使用 MM:SS
- 測試 8: 自動時間格式 (短音訊 <60s) 應使用 SECONDS

最新執行結果: tests/results/test_enhanced_pipeline_latest.txt
參考結果: tests/results/test_enhanced_pipeline_reference.txt
"""

import torch
from test_utils import TestResultSaver, load_audio, get_test_audio_path
from omnilingual_asr.enhanced_pipeline import (
    EnhancedASRPipeline,
    TimestampFormat,
    TimeFormat
)


def test_enhanced_pipeline():
    """測試增強版 Pipeline"""
    
    audio_path = get_test_audio_path()
    
    if not audio_path.exists():
        print(f"❌ 找不到音訊檔案: {audio_path}")
        return
    
    # 載入音訊
    waveform, sr = load_audio(str(audio_path))
    audio_duration = len(waveform) / sr
    print(f"音訊長度: {audio_duration:.2f} 秒")
    print()
    
    # 初始化 Pipeline
    pipeline = EnhancedASRPipeline(
        model_card="omniASR_LLM_3B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    inp = [{
        "waveform": waveform,
        "sample_rate": sr
    }]
    
    # 測試 1: 無時間戳記
    print("=" * 70)
    print("測試 1: 無時間戳記")
    print("=" * 70)
    result = pipeline.transcribe(
        inp=inp,
        batch_size=1,
        lang=["cmn_Hant"],
        chunk_duration=30.0,
        overlap=1.0,
        timestamp_format=TimestampFormat.NONE
    )
    print(result[0])
    print()
    
    # 測試 2: 簡易時間戳記 (MM:SS)
    print("=" * 70)
    print("測試 2: 簡易時間戳記 (MM:SS)")
    print("=" * 70)
    result = pipeline.transcribe(
        inp=inp,
        batch_size=1,
        lang=["cmn_Hant"],
        chunk_duration=30.0,
        overlap=1.0,
        timestamp_format=TimestampFormat.SIMPLE,
        time_format=TimeFormat.MMSS
    )
    print(result[0])
    print()
    
    # 測試 3: 詳細時間戳記 (HH:MM:SS)
    print("=" * 70)
    print("測試 3: 詳細時間戳記 (HH:MM:SS)")
    print("=" * 70)
    result = pipeline.transcribe(
        inp=inp,
        batch_size=1,
        lang=["cmn_Hant"],
        chunk_duration=30.0,
        overlap=1.0,
        timestamp_format=TimestampFormat.DETAILED,
        time_format=TimeFormat.HHMMSS
    )
    print(result[0])
    print()
    
    # 測試 4: 自訂時間戳記模板
    print("=" * 70)
    print("測試 4: 自訂時間戳記模板")
    print("=" * 70)
    result = pipeline.transcribe(
        inp=inp,
        batch_size=1,
        lang=["cmn_Hant"],
        chunk_duration=30.0,
        overlap=1.0,
        timestamp_format=TimestampFormat.SIMPLE,
        time_format=TimeFormat.SECONDS,
        timestamp_template="⏱️ {start} | {text}"
    )
    print(result[0])
    print()
    
    # 測試 5: JSON 格式
    print("=" * 70)
    print("測試 5: JSON 格式")
    print("=" * 70)
    result = pipeline.transcribe(
        inp=inp,
        batch_size=1,
        lang=["cmn_Hant"],
        chunk_duration=30.0,
        overlap=1.0,
        timestamp_format=TimestampFormat.JSON
    )
    import json
    print(json.dumps(result[0], indent=2, ensure_ascii=False))
    
    # Verify dynamic chunking
    print("\n[VAD 切割驗證] 檢查片段長度是否動態調整:")
    for i, chunk in enumerate(result[0]):
        duration = chunk["end"] - chunk["start"]
        print(f"  Chunk {i+1}: {duration:.2f}s (Start: {chunk['start']:.2f}s, End: {chunk['end']:.2f}s)")
    print()
    
    # 測試 6: 錯誤處理 - chunk_duration 超過限制
    print("=" * 70)
    print("測試 6: 錯誤處理 - chunk_duration 超過限制")
    print("=" * 70)
    try:
        result = pipeline.transcribe(
            inp=inp,
            batch_size=1,
            lang=["cmn_Hant"],
            chunk_duration=50.0,  # 超過 40 秒限制
            overlap=1.0
        )
    except ValueError as e:
        print(f"✓ 正確捕獲錯誤: {e}")
    print()

    # 測試 7: 自動時間格式 (長音訊 > 60s)
    print("=" * 70)
    print("測試 7: 自動時間格式 (長音訊 > 60s)")
    print("=" * 70)
    print("預期: 應使用 MM:SS 格式")
    result = pipeline.transcribe(
        inp=inp,
        batch_size=1,
        lang=["cmn_Hant"],
        chunk_duration=30.0,
        overlap=1.0,
        timestamp_format=TimestampFormat.SIMPLE,
        time_format=TimeFormat.AUTO
    )
    print(result[0])
    print()

    # 測試 8: 自動時間格式 (短音訊 < 60s)
    print("=" * 70)
    print("測試 8: 自動時間格式 (短音訊 < 60s)")
    print("=" * 70)
    print("預期: 應使用 SECONDS 格式")
    
    # 創建一個短音訊片段
    short_waveform = waveform[:int(30 * sr)]
    short_inp = [{
        "waveform": short_waveform,
        "sample_rate": sr
    }]
    
    result = pipeline.transcribe(
        inp=short_inp,
        batch_size=1,
        lang=["cmn_Hant"],
        chunk_duration=30.0,
        overlap=1.0,
        timestamp_format=TimestampFormat.SIMPLE,
        time_format=TimeFormat.AUTO
    )
    print(result[0])
    print()


if __name__ == "__main__":
    with TestResultSaver("test_enhanced_pipeline"):
        test_enhanced_pipeline()