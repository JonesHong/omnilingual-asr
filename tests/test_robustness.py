#!/usr/bin/env python3
"""
測試 StreamingASRPipeline 的穩健性（記憶體洩漏、長時間運行）

執行結果會自動保存在 tests/results/ 目錄下。

預期行為：
- 測試 1: 靜音輸入不應導致崩潰
- 測試 2: 長時間運行不應有明顯記憶體洩漏（< 100MB 增長）

最新執行結果: tests/results/test_robustness_latest.txt
參考結果: tests/results/test_robustness_reference.txt
"""

import torch
import torchaudio
import time
import psutil
import os
import numpy as np
from test_utils import TestResultSaver, get_test_audio_path
from omnilingual_asr.streaming import StreamingASRPipeline


def get_memory_usage():
    """獲取當前進程的記憶體使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_robustness(model_card="omniASR_CTC_300M"):
    print(f"Robustness Testing with {model_card}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipeline = StreamingASRPipeline(
        model_card=model_card,
        device=device,
        dtype=dtype
    )
    
    # 1. Silence Test
    print("\n[Test 1] Silence Input (10 seconds)")
    sr = 16000
    silence_chunk = np.zeros(int(0.5 * sr), dtype=np.float32)
    
    start_mem = get_memory_usage()
    for _ in range(20): # 10 seconds
        pipeline.add_audio(silence_chunk)
        list(pipeline.transcribe_available())
    list(pipeline.finish())
    end_mem = get_memory_usage()
    print(f"Memory change: {end_mem - start_mem:.2f} MB")
    
    # 2. Long Running Simulation (Memory Leak Check)
    print("\n[Test 2] Long Running Simulation (Repeating Audio)")
    
    # Load real audio
    audio_path = get_test_audio_path()
    waveform, _ = torchaudio.load(str(audio_path))
    audio = waveform.squeeze(0).numpy()
    if waveform.shape[0] > 1:
        audio = waveform.mean(dim=0).numpy()
        
    chunk_size = int(0.5 * sr)
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
    
    # Repeat audio 5 times to simulate longer session
    print(f"Processing {len(chunks) * 5} chunks...")
    
    mem_usages = []
    start_time = time.time()
    
    for i in range(5):
        for chunk in chunks:
            pipeline.add_audio(chunk)
            list(pipeline.transcribe_available())
            
            if len(mem_usages) % 100 == 0:
                mem_usages.append(get_memory_usage())
    
    list(pipeline.finish())
    end_time = time.time()
    
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Initial Memory: {mem_usages[0]:.2f} MB")
    print(f"Final Memory: {get_memory_usage():.2f} MB")
    
    # Check for significant increase
    if get_memory_usage() - mem_usages[0] > 100: # > 100MB increase
        print("WARNING: Potential memory leak detected!")
    else:
        print("Memory usage stable.")


if __name__ == "__main__":
    with TestResultSaver("test_robustness"):
        test_robustness()