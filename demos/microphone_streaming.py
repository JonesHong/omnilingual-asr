
import sys
import time
import numpy as np
import torch
import queue
from omnilingual_asr.streaming import StreamingASRPipeline

# Try to import sounddevice
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(1)

def main():
    # Configuration
    MODEL_CARD = "omniASR_CTC_300M"
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 4000 # 0.25s chunks from mic
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Initializing Streaming Pipeline with {MODEL_CARD}...")
    pipeline = StreamingASRPipeline(
        model_card=MODEL_CARD,
        chunk_duration=3.0,
        stride_left=0.5,
        stride_right=0.5,
        device=DEVICE,
        dtype=DTYPE
    )

    # Queue to communicate between audio callback and main thread
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # indata is (frames, channels). We want mono.
        # Make sure to copy() to avoid issues with buffer reuse
        q.put(indata.copy())

    print("\n" + "="*50)
    print(f"Listening... (Press Ctrl+C to stop)")
    print("="*50)

    try:
        # Open microphone stream
        # channels=1 (Mono), samplerate=16000
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                            device=None, channels=1, callback=callback):
            
            while True:
                # 1. Get audio from queue
                try:
                    # Non-blocking check or short timeout
                    indata = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 2. Process audio
                # indata is [frames, 1], flatten to [frames]
                audio_chunk = indata.flatten().astype(np.float32)
                
                pipeline.add_audio(audio_chunk)
                
                # 3. Get results
                for result in pipeline.transcribe_available():
                    # Print with carriage return to simulate streaming update?
                    # Or just print new segments.
                    # Since we only get "middle" text which is stable, we can just print it.
                    print(f"{result.text}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        # Process remaining
        for result in pipeline.finish():
            print(f"{result.text}", end="", flush=True)
        print("\nDone.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
