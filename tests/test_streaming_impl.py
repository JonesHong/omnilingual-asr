
import torch
import torchaudio
import time
from pathlib import Path
import numpy as np
from omnilingual_asr.streaming import StreamingASRPipeline

def load_audio(audio_path: str, target_sr: int = 16000):
    """Load audio and resample."""
    print(f"Loading audio: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy(), target_sr

def test_streaming(audio_path: str, model_card="omniASR_CTC_300M"):
    print(f"Testing Streaming ASR with {model_card}...")
    
    # Initialize Pipeline
    pipeline = StreamingASRPipeline(
        model_card=model_card,
        chunk_duration=3.0,
        stride_left=0.5,
        stride_right=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load Audio
    audio, sr = load_audio(audio_path)
    print(f"Audio length: {len(audio)/sr:.2f}s")
    
    # Simulate Streaming
    chunk_size = int(0.5 * sr) # Feed 0.5s at a time
    cursor = 0
    
    print("\nStarting Stream...")
    start_time = time.time()
    full_text = ""
    
    while cursor < len(audio):
        chunk = audio[cursor : cursor + chunk_size]
        cursor += chunk_size
        
        pipeline.add_audio(chunk)
        
        for result in pipeline.transcribe_available():
            print(f"[{result.timestamp:.2f}s] {result.text}")
            full_text += result.text
            
    # Finish
    for result in pipeline.finish():
        print(f"[{result.timestamp:.2f}s] {result.text}")
        full_text += result.text
        
    total_time = time.time() - start_time
    print(f"\nTotal Time: {total_time:.2f}s")
    print(f"Full Text: {full_text}")

if __name__ == "__main__":
    # Use relative path to find the audio file
    project_root = Path(__file__).parent.parent
    audio_path = project_root / "什麼是上帝的道.mp3"
    
    if audio_path.exists():
        test_streaming(str(audio_path))
    else:
        print(f"Audio file not found: {audio_path}")
