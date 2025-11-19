
import torch
import torchaudio
import time
import numpy as np
from pathlib import Path
from omnilingual_asr.streaming import StreamingASRPipeline

def load_audio(audio_path: str, target_sr: int = 16000):
    """Load audio and resample."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy(), target_sr

def benchmark_streaming(audio_path: str, model_card="omniASR_CTC_300M"):
    print(f"Benchmarking Streaming ASR with {model_card}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Initialize Pipeline
    init_start = time.time()
    pipeline = StreamingASRPipeline(
        model_card=model_card,
        chunk_duration=3.0,
        stride_left=0.5,
        stride_right=0.5,
        device=device,
        dtype=dtype
    )
    init_time = time.time() - init_start
    print(f"Initialization Time: {init_time:.2f}s")
    
    # Load Audio
    audio, sr = load_audio(audio_path)
    audio_duration = len(audio) / sr
    print(f"Audio length: {audio_duration:.2f}s")
    
    # Simulate Streaming
    chunk_size_sec = 0.5
    chunk_size = int(chunk_size_sec * sr) 
    cursor = 0
    
    latencies = []
    processing_times = []
    
    print("\nStarting Benchmark Stream...")
    start_time = time.time()
    
    while cursor < len(audio):
        chunk_start_time = time.time()
        
        # Simulate real-time arrival (optional, but good for realistic latency)
        # time.sleep(chunk_size_sec) 
        
        chunk = audio[cursor : cursor + chunk_size]
        cursor += chunk_size
        
        pipeline.add_audio(chunk)
        
        # Process
        proc_start = time.time()
        results = list(pipeline.transcribe_available())
        proc_end = time.time()
        
        proc_time = proc_end - proc_start
        processing_times.append(proc_time)
        
        if results:
            # Latency = Time result available - Time audio chunk end (simulated)
            # Here we just measure processing time as a proxy for added latency
            # Real latency would include the chunk accumulation time.
            pass
            
    # Finish
    proc_start = time.time()
    list(pipeline.finish())
    proc_end = time.time()
    processing_times.append(proc_end - proc_start)
        
    total_time = time.time() - start_time
    
    # Metrics
    avg_proc_time = np.mean(processing_times)
    p90_proc_time = np.percentile(processing_times, 90)
    max_proc_time = np.max(processing_times)
    
    rtf = total_time / audio_duration
    
    print("\n" + "="*40)
    print("Benchmark Results")
    print("="*40)
    print(f"Total Processing Time: {total_time:.2f}s")
    print(f"Audio Duration: {audio_duration:.2f}s")
    print(f"Real Time Factor (RTF): {rtf:.3f}")
    print(f"Speedup: {1/rtf:.2f}x")
    print("-" * 20)
    print(f"Avg Processing Time per {chunk_size_sec}s chunk: {avg_proc_time*1000:.2f}ms")
    print(f"P90 Processing Time: {p90_proc_time*1000:.2f}ms")
    print(f"Max Processing Time: {max_proc_time*1000:.2f}ms")
    print("="*40)

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    audio_path = project_root / "什麼是上帝的道.mp3"
    
    if audio_path.exists():
        benchmark_streaming(str(audio_path))
    else:
        print(f"Audio file not found: {audio_path}")
