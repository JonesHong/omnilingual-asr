#!/usr/bin/env python3
"""
測試 StreamingASRPipeline 的雙層緩衝機制（Two-Pass Strategy）

執行結果會自動保存在 tests/results/ 目錄下。

預期行為：
- [PREVIEW]: 快速路徑，低延遲預覽結果（is_stable=False）
- [STABLE]: 慢速路徑，長上下文修正結果（is_stable=True）
- 最終穩定文本應該比預覽文本更準確

最新執行結果: tests/results/test_streaming_latest.txt
參考結果: tests/results/test_streaming_reference.txt
"""

import torch
import time
import numpy as np
from test_utils import TestResultSaver, load_audio, get_test_audio_path
from omnilingual_asr.streaming import StreamingASRPipeline


def test_streaming(audio_path: str, model_card="omniASR_LLM_300M", lang="cmn_Hant"):
    print(f"Testing Two-Pass Streaming ASR with {model_card}...")
    
    # Initialize Pipeline with 10s context for faster testing
    # Note: We are using the unified StreamingASRPipeline now
    pipeline = StreamingASRPipeline(
        model_card=model_card,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        lang=lang,
        use_vad=True,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        enable_stability_pass=True,  # Explicitly enable two-pass strategy
        context_duration_ms=10000  # 10 seconds for stable correction
    )
    
    # Load Audio
    audio, sr = load_audio(audio_path)
    print(f"Audio length: {len(audio)/sr:.2f}s")
    
    # Simulate Streaming
    chunk_size = int(0.5 * sr) # Feed 0.5s at a time
    cursor = 0
    
    print("\nStarting Stream...")
    print("-" * 80)
    print(f"{'Type':<10} | {'Time Range':<20} | {'Text'}")
    print("-" * 80)
    
    start_time = time.time()
    stable_text_accumulator = ""
    
    def process_result(result):
        nonlocal stable_text_accumulator
        
        tag = "[STABLE]" if result.is_stable else "[PREVIEW]"
        time_range = f"{result.start_timestamp:.1f}s - {result.timestamp:.1f}s"
        
        # Visual distinction
        if result.is_stable:
            print(f"\033[92m{tag:<10} | {time_range:<20} | {result.text}\033[0m") # Green for stable
            stable_text_accumulator += result.text + " "
        else:
            print(f"{tag:<10} | {time_range:<20} | {result.text}")

    while cursor < len(audio):
        chunk = audio[cursor : cursor + chunk_size]
        cursor += chunk_size
        
        pipeline.add_audio(chunk)
        
        for result in pipeline.transcribe_available():
            process_result(result)
            
    # Process remaining audio
    print("\nFinishing stream...")
    for result in pipeline.finish():
        process_result(result)
        
    total_time = time.time() - start_time
    print("-" * 80)
    print(f"\nTotal Time: {total_time:.2f}s")
    print(f"Final Stable Text: {stable_text_accumulator.strip()}")


if __name__ == "__main__":
    audio_path = get_test_audio_path()
    
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
    else:
        with TestResultSaver("test_streaming"):
            # Test with LLM model
            test_streaming(str(audio_path), model_card="omniASR_LLM_300M", lang="cmn_Hant")
