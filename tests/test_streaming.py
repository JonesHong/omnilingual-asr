import unittest
from unittest.mock import MagicMock, patch
import sys
import types

# Helper to mock packages
def mock_package(name):
    m = types.ModuleType(name)
    m.__path__ = [] # Mark as package
    sys.modules[name] = m
    return m

# Mock fairseq2 hierarchy
fairseq2 = mock_package("fairseq2")
fairseq2.assets = mock_package("fairseq2.assets") # Sometimes used

fairseq2.nn = mock_package("fairseq2.nn")
fairseq2.nn.batch_layout = mock_package("fairseq2.nn.batch_layout")
fairseq2.nn.batch_layout.BatchLayout = MagicMock()

fairseq2.data = mock_package("fairseq2.data")
fairseq2.data.tokenizers = mock_package("fairseq2.data.tokenizers")
fairseq2.data.tokenizers.Tokenizer = MagicMock()
fairseq2.data.audio = mock_package("fairseq2.data.audio")
fairseq2.data.audio.AudioDecoder = MagicMock()

fairseq2.data.data_pipeline = mock_package("fairseq2.data.data_pipeline")
fairseq2.data.data_pipeline.CollateOptionsOverride = MagicMock()
fairseq2.data.data_pipeline.Collater = MagicMock()
fairseq2.data.data_pipeline.DataPipeline = MagicMock()
fairseq2.data.data_pipeline.DataPipelineBuilder = MagicMock()
fairseq2.data.data_pipeline.FileMapper = MagicMock()
fairseq2.data.data_pipeline.read_sequence = MagicMock()

fairseq2.data._memory = mock_package("fairseq2.data._memory")
fairseq2.data._memory.MemoryBlock = MagicMock()

fairseq2.datasets = mock_package("fairseq2.datasets")
fairseq2.datasets.batch = mock_package("fairseq2.datasets.batch")
fairseq2.datasets.batch.Seq2SeqBatch = MagicMock()

fairseq2.logging = mock_package("fairseq2.logging")
fairseq2.logging.get_log_writer = MagicMock()

fairseq2.models = mock_package("fairseq2.models")
fairseq2.models.hub = mock_package("fairseq2.models.hub")
fairseq2.models.hub.load_model = MagicMock()
fairseq2.models.hub.load_tokenizer = MagicMock()

fairseq2.models.wav2vec2 = mock_package("fairseq2.models.wav2vec2")
fairseq2.models.wav2vec2.asr = mock_package("fairseq2.models.wav2vec2.asr")
fairseq2.models.wav2vec2.asr.Wav2Vec2AsrModel = MagicMock()

# Mock fairseq2.composition
mock_package("fairseq2.composition")
comp_assets = mock_package("fairseq2.composition.assets")
comp_assets.register_package_assets = MagicMock()

# Mock pipeline
pipeline = mock_package("omnilingual_asr.models.inference.pipeline")
pipeline.ASRInferencePipeline = MagicMock()

import numpy as np
import torch
from omnilingual_asr.models.inference.streaming import StrideAudioBuffer, CTCAlignmentExtractor

class TestStrideAudioBuffer(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.chunk_duration = 1.0 # Shorten for test
        self.stride_left = 0.2
        self.stride_right = 0.2
        self.buffer = StrideAudioBuffer(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            stride_left=self.stride_left,
            stride_right=self.stride_right,
            use_vad=False # Disable VAD for logic test
        )

    def test_buffer_accumulation(self):
        # Add 0.5s audio
        audio = np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
        self.buffer.add_audio(audio)
        self.assertFalse(self.buffer.has_chunk())
        
        # Add another 0.5s
        self.buffer.add_audio(audio)
        self.assertTrue(self.buffer.has_chunk())

    def test_stride_advancement(self):
        # Chunk size = 16000
        # Keep size = 16000 - 3200 - 3200 = 9600 (0.6s)
        
        # Add 2.0s audio
        audio = np.arange(int(2.0 * self.sample_rate), dtype=np.float32)
        self.buffer.add_audio(audio)
        
        # First chunk
        chunk1, _ = self.buffer.get_chunk()
        self.assertEqual(len(chunk1), 16000)
        self.assertTrue(np.array_equal(chunk1, audio[:16000]))
        
        # Buffer should have advanced by keep_size (9600)
        # Remaining buffer start should be at index 9600 of original audio
        # Current buffer length should be 32000 - 9600 = 22400
        self.assertEqual(len(self.buffer.buffer), 22400)
        self.assertEqual(self.buffer.buffer[0], audio[9600])
        
        # Second chunk
        chunk2, _ = self.buffer.get_chunk()
        self.assertEqual(len(chunk2), 16000)
        # Chunk 2 should start at 9600
        self.assertTrue(np.array_equal(chunk2, audio[9600:9600+16000]))

class TestCTCAlignmentExtractor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.decode.return_value = "test"
        self.extractor = CTCAlignmentExtractor(self.tokenizer, blank_id=0)

    def test_extract_middle(self):
        # 50 frames total
        # Stride left 10, right 10
        # Keep 10-40
        logits = torch.zeros(50, 10)
        # Set some values
        # Frame 20: token 1
        # Frame 25: token 2
        logits[20, 1] = 10.0
        logits[25, 2] = 10.0
        
        text = self.extractor.extract_middle(logits, 10, 10)
        
        # Verify tokenizer called with correct tokens
        # Expected: [1, 2] (after collapse)
        # The mock decode just returns "test", but we can check call args
        call_args = self.tokenizer.decode.call_args[0][0]
        self.assertTrue(torch.equal(call_args, torch.tensor([1, 2])))

    def test_extract_middle_boundary(self):
        # Test tokens exactly at boundary
        logits = torch.zeros(50, 10)
        # Frame 9: token 1 (should be skipped, < 10)
        # Frame 10: token 2 (should be kept)
        # Frame 39: token 3 (should be kept)
        # Frame 40: token 4 (should be skipped, >= 40)
        
        logits[9, 1] = 10.0
        logits[10, 2] = 10.0
        logits[39, 3] = 10.0
        logits[40, 4] = 10.0
        
        self.extractor.extract_middle(logits, 10, 10)
        
        call_args = self.tokenizer.decode.call_args[0][0]
        self.assertTrue(torch.equal(call_args, torch.tensor([2, 3])))

if __name__ == '__main__':
    unittest.main()
