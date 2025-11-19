#!/usr/bin/env python3
"""
測試 EnhancedASRPipeline 的音訊切割與時間戳記功能
"""

import torch
import torchaudio
from pathlib import Path
from omnilingual_asr.enhanced_pipeline import (
    EnhancedASRPipeline,
    TimestampFormat,
    TimeFormat
)


def load_audio(audio_path: str, target_sr: int = 16000):
    """載入音訊"""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0), target_sr


def test_enhanced_pipeline():
    """測試增強版 Pipeline"""
    
    audio_path = "/mnt/c/work/omnilingual-asr/什麼是上帝的道.mp3"
    
    if not Path(audio_path).exists():
        print(f"❌ 找不到音訊檔案: {audio_path}")
        return
    
    # 載入音訊
    waveform, sr = load_audio(audio_path)
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
        timestamp_format=TimestampFormat.SIMPLE,  # 需要非 NONE
        time_format=TimeFormat.SECONDS,
        timestamp_template="⏱️ {start} | {text}"
    )
    print(result[0])
    print()
    
    # 測試 5: 錯誤處理 - chunk_duration 超過限制
    print("=" * 70)
    print("測試 5: 錯誤處理 - chunk_duration 超過限制")
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


if __name__ == "__main__":
    test_enhanced_pipeline()
