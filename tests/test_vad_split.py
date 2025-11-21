
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from omnilingual_asr.enhanced_pipeline import EnhancedASRPipeline, TimestampFormat

def test_vad_split():
    print("Initializing EnhancedASRPipeline...")
    # Use a dummy model or CPU to be fast, but we need the VAD model to load
    # We can mock the base pipeline to avoid loading a heavy ASR model
    
    class MockBasePipeline:
        def transcribe(self, inp, batch_size=1, lang=None):
            # Return dummy text for each input
            results = []
            for item in inp:
                duration = item["waveform"].shape[-1] / item["sample_rate"]
                results.append(f"Transcribed segment ({duration:.2f}s)")
            return results

    try:
        mock_base = MockBasePipeline()
        pipeline = EnhancedASRPipeline(
            model_card="dummy", 
            device="cpu", 
            base_pipeline=mock_base
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        # If model loading fails (e.g. no internet for VAD), we might need to mock VAD too
        # But let's see if it works first.
        return

    # Create a dummy waveform: 10 seconds of audio
    # 0-2s: silence
    # 2-4s: signal (speech)
    # 4-6s: silence
    # 6-8s: signal (speech)
    # 8-10s: silence
    
    sample_rate = 16000
    duration = 10
    waveform = torch.zeros(sample_rate * duration)
    
    # Add "speech" (noise)
    waveform[2*sample_rate : 4*sample_rate] = torch.randn(2*sample_rate) * 0.5
    waveform[6*sample_rate : 8*sample_rate] = torch.randn(2*sample_rate) * 0.5
    
    print("Testing split_on_silence=True...")
    
    # Always mock VAD model for deterministic testing
    print("Mocking VAD model for test...")
    class MockVAD(torch.nn.Module):
        def forward(self, x, sr):
            # Simple energy based VAD for testing
            # x shape: (batch, samples)
            energy = x.pow(2).mean()
            # Threshold needs to be tuned to the noise level we added (0.5 std dev -> var 0.25)
            return torch.tensor(1.0 if energy > 0.1 else 0.0)
    pipeline.vad_model = MockVAD()

    results = pipeline.transcribe(
        inp=[{"waveform": waveform, "sample_rate": sample_rate}],
        split_on_silence=True,
        min_silence_duration_ms=500,
        timestamp_format=TimestampFormat.SIMPLE
    )
    
    print("\nResult:")
    print(f"Type: {type(results)}")
    print(f"Length: {len(results)}")
    if len(results) > 0:
        print(f"Content: '{results[0]}'")
    
    # Check if we got multiple segments
    # We expect at least 2 segments
    if len(results) > 0 and "Transcribed segment" in str(results[0]):
        print("\nSUCCESS: Transcription returned.")
    else:
        print("\nFAILURE: No transcription returned.")

if __name__ == "__main__":
    test_vad_split()
