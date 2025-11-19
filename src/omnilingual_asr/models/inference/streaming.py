from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torchaudio.functional as F
from fairseq2.nn.batch_layout import BatchLayout
from fairseq2.data.tokenizers import Tokenizer

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

log = logging.getLogger(__name__)

@dataclass
class StreamingResult:
    text: str
    is_final: bool
    timestamp: float
    latency: float

class StrideAudioBuffer:
    """
    Manages audio buffer for stride-based streaming.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        stride_left: float = 0.5,
        stride_right: float = 0.5,
        use_vad: bool = True,
        vad_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.stride_left = stride_left
        self.stride_right = stride_right
        self.keep_duration = chunk_duration - stride_left - stride_right
        
        if self.keep_duration <= 0:
            raise ValueError("keep_duration must be positive. Check chunk_duration and strides.")

        self.chunk_size = int(chunk_duration * sample_rate)
        self.stride_left_size = int(stride_left * sample_rate)
        self.stride_right_size = int(stride_right * sample_rate)
        self.keep_size = self.chunk_size - self.stride_left_size - self.stride_right_size
        self.stride_step = self.keep_size # We advance by the kept amount

        self.buffer = deque()
        self.use_vad = use_vad
        self.vad_threshold = vad_threshold
        self.vad_model = None

        if use_vad:
            self._init_vad()

    def _init_vad(self):
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self.vad_model.eval()
        except Exception as e:
            log.warning(f"VAD initialization failed: {e}. VAD will be disabled.")
            self.use_vad = False

    def add_audio(self, audio: np.ndarray) -> None:
        """Add audio to buffer. Audio should be 16kHz mono float32."""
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        
        self.buffer.extend(audio.astype(np.float32))

    def has_chunk(self) -> bool:
        return len(self.buffer) >= self.chunk_size

    def get_chunk(self) -> Optional[Tuple[np.ndarray, bool]]:
        """
        Returns (chunk, is_speech)
        chunk: np.ndarray of size chunk_size
        is_speech: bool indicating if speech was detected in the *middle* part
        """
        if not self.has_chunk():
            return None

        # Extract chunk without removing from buffer yet (we need overlap)
        # But wait, deque doesn't support slicing easily.
        # We can convert to list, but that's slow.
        # Optimization: maintain a separate numpy buffer or just pay the cost for now.
        # Given 3s @ 16kHz = 48k samples, list conversion is acceptable but not ideal.
        # Let's use itertools.islice
        from itertools import islice
        chunk = np.array(list(islice(self.buffer, 0, self.chunk_size)), dtype=np.float32)

        is_speech = True
        if self.use_vad and self.vad_model is not None:
            # Check VAD on the middle part (the part we keep)
            middle_part = chunk[self.stride_left_size : self.chunk_size - self.stride_right_size]
            if len(middle_part) > 0:
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(middle_part).float()
                    # Silero expects [Batch, Time] or [Time]
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                    is_speech = speech_prob > self.vad_threshold

        # Advance buffer by stride_step (keep_size)
        # We remove the part that we have "processed" and "kept"
        # The stride logic in the spec says:
        # "Chunk 1: [0.5 | 2.0 | 0.5]"
        # "Chunk 2:       [0.5 | 2.0 | 0.5]"
        # So we advance by 2.0s (keep_size).
        # But wait, we need to be careful.
        # If we just popleft keep_size, the next chunk starts at (old_start + keep_size).
        # Chunk 1 start: 0.0. Middle: 0.5-2.5.
        # Chunk 2 start: 2.0. Middle: 2.5-4.5.
        # Yes, this aligns.
        
        for _ in range(self.keep_size):
            if self.buffer:
                self.buffer.popleft()

        return chunk, is_speech

    def clear(self):
        self.buffer.clear()


class CTCAlignmentExtractor:
    """
    Extracts text from CTC logits using alignment constraints.
    """
    def __init__(self, tokenizer: Tokenizer, blank_id: int = 0, frames_per_second: int = 50):
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.frames_per_second = frames_per_second

    def extract_middle(
        self,
        logits: torch.Tensor,
        stride_left_frames: int,
        stride_right_frames: int,
    ) -> str:
        """
        logits: [T, Vocab]
        """
        pred_ids = torch.argmax(logits, dim=-1)
        total_frames = pred_ids.size(0)
        
        keep_start = stride_left_frames
        keep_end = total_frames - stride_right_frames
        
        if keep_start >= keep_end:
            return ""
            
        middle_ids = pred_ids[keep_start:keep_end]
        
        # CTC Collapse
        if len(middle_ids) == 0:
            return ""
            
        # 1. Remove duplicates
        unique_mask = torch.ones(len(middle_ids), dtype=torch.bool, device=middle_ids.device)
        unique_mask[1:] = middle_ids[1:] != middle_ids[:-1]
        collapsed = middle_ids[unique_mask]
        
        # 2. Remove blanks
        non_blank = collapsed[collapsed != self.blank_id]
        
        if len(non_blank) == 0:
            return ""
            
        return self.tokenizer.decode(non_blank)


class StreamingASRPipeline:
    def __init__(
        self,
        model_card: str = "omniASR_CTC_1B",
        chunk_duration: float = 3.0,
        stride_left: float = 0.5,
        stride_right: float = 0.5,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lang: Optional[str] = None,
        use_vad: bool = True
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.lang = lang
        
        # Initialize base pipeline
        self.base_pipeline = ASRInferencePipeline(
            model_card=model_card,
            device=device,
            dtype=dtype
        )
        
        self.audio_buffer = StrideAudioBuffer(
            sample_rate=16000,
            chunk_duration=chunk_duration,
            stride_left=stride_left,
            stride_right=stride_right,
            use_vad=use_vad
        )
        
        # Initialize CTC Extractor
        # We need to find blank_id. Usually it's 0 in fairseq2, but let's check vocab info if possible.
        # For now assume 0.
        self.ctc_extractor = CTCAlignmentExtractor(
            tokenizer=self.base_pipeline.token_decoder,
            blank_id=0,
            frames_per_second=50 # Wav2Vec2 standard
        )
        
        self.current_time = 0.0
        self.stride_left_frames = int(stride_left * 50)
        self.stride_right_frames = int(stride_right * 50)

    def add_audio(self, audio: np.ndarray):
        self.audio_buffer.add_audio(audio)

    def _process_chunk_optimized(self, audio_chunk: np.ndarray) -> str:
        # 1. Prepare Tensor
        waveform = torch.from_numpy(audio_chunk).float().to(self.device)
        if self.dtype == torch.float16:
            waveform = waveform.half()
        elif self.dtype == torch.bfloat16:
            waveform = waveform.bfloat16()
        
        # 2. Normalize (Zero-mean, Unit-variance)
        with torch.no_grad():
            mean = waveform.mean()
            std = waveform.std()
            if std == 0:
                std = 1.0
            waveform = (waveform - mean) / std

        # 3. Batch Layout
        seq_lens = torch.tensor([waveform.size(0)], device=self.device)
        batch_layout = BatchLayout(
            waveform.unsqueeze(0), # [1, T]
            seq_lens=seq_lens,
            device=self.device
        )

        # 4. Model Forward
        with torch.inference_mode():
            # Wav2Vec2AsrModel returns (logits, padding_mask)
            # We access the model directly. 
            # base_pipeline.model is a Wav2Vec2AsrModel or Wav2Vec2LlamaModel.
            # The spec assumes CTC model (Wav2Vec2AsrModel).
            
            # Note: fairseq2 models usually return a named tuple or similar.
            # Wav2Vec2AsrModel forward returns: (logits, padding_mask)
            output = self.base_pipeline.model(waveform.unsqueeze(0), batch_layout)
            
            # Handle different return types if necessary, but for Wav2Vec2AsrModel it should be tuple
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

        # 5. Extract Text
        return self.ctc_extractor.extract_middle(
            logits[0], 
            self.stride_left_frames, 
            self.stride_right_frames
        )

    def transcribe_available(self) -> Iterator[StreamingResult]:
        while self.audio_buffer.has_chunk():
            chunk_data = self.audio_buffer.get_chunk()
            if chunk_data is None:
                break
                
            audio_chunk, is_speech = chunk_data
            
            text = ""
            if is_speech:
                import time
                start_t = time.time()
                text = self._process_chunk_optimized(audio_chunk)
                latency = time.time() - start_t
            else:
                latency = 0.0
                
            # Update timestamp
            # The timestamp should reflect the end of the *kept* audio
            self.current_time += self.audio_buffer.keep_duration
            
            if text or is_speech: # Yield even if empty text if speech was detected, or maybe only if text?
                # Spec says: "StreamingResult"
                yield StreamingResult(
                    text=text,
                    is_final=False, # Stride based is never "final" in the sense of sentence end, unless VAD says so.
                    # But here we just output chunks.
                    timestamp=self.current_time,
                    latency=latency
                )

    def finish(self) -> Iterator[StreamingResult]:
        # Flush remaining audio?
        # The buffer might have some audio left that is < chunk_size.
        # We can pad it to chunk_size and process.
        remaining_len = len(self.audio_buffer.buffer)
        if remaining_len > 0:
            padding = np.zeros(self.audio_buffer.chunk_size - remaining_len, dtype=np.float32)
            self.audio_buffer.add_audio(padding)
            yield from self.transcribe_available()
