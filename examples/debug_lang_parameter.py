#!/usr/bin/env python3
"""
測試 lang 參數對 LLM 模型輸出的影響
"""
import torch
import torchaudio
from pathlib import Path
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

def test_lang_effect():
    audio_path = Path("/mnt/c/work/omnilingual-asr/什麼是上帝的道.mp3")
    
    # Load a short segment (first 5 seconds)
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Take first 5 seconds
    segment = waveform.squeeze(0)[:5*sr]
    
    # Initialize pipeline
    pipeline = ASRInferencePipeline(
        model_card="omniASR_LLM_3B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    inp = [{
        "waveform": segment,
        "sample_rate": sr,
    }]
    
    print("=" * 70)
    print("Test 1: WITHOUT lang parameter")
    print("=" * 70)
    result1 = pipeline.transcribe(inp=inp, batch_size=1, lang=None)
    print(f"Result: {result1[0]}")
    print()
    
    print("=" * 70)
    print("Test 2: WITH lang='cmn_Hant'")
    print("=" * 70)
    result2 = pipeline.transcribe(inp=inp, batch_size=1, lang=["cmn_Hant"])
    print(f"Result: {result2[0]}")
    print()
    
    print("=" * 70)
    print("Test 3: WITH lang='cmn_Hans'")
    print("=" * 70)
    result3 = pipeline.transcribe(inp=inp, batch_size=1, lang=["cmn_Hans"])
    print(f"Result: {result3[0]}")
    print()
    
    print("=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"No lang:   {len(result1[0])} chars")
    print(f"cmn_Hant:  {len(result2[0])} chars")
    print(f"cmn_Hans:  {len(result3[0])} chars")

if __name__ == "__main__":
    test_lang_effect()
