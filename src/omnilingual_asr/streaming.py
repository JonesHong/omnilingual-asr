# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from fairseq2.nn.batch_layout import BatchLayout
from torch.nn.functional import layer_norm

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


@dataclass
class StreamingResult:
    text: str          # The text recognized in this chunk
    is_final: bool     # Whether this result is final (usually for VAD segmentation)
    timestamp: float   # The end timestamp of this text in the stream
    latency: float     # Processing latency


class StrideAudioBuffer:
    def __init__(
        self,
        chunk_duration: float = 3.0,
        stride_left: float = 0.5,
        stride_right: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.chunk_duration = chunk_duration
        self.stride_left = stride_left
        self.stride_right = stride_right
        self.sample_rate = sample_rate
        
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.stride_left_samples = int(stride_left * sample_rate)
        self.stride_right_samples = int(stride_right * sample_rate)
        self.keep_samples = self.chunk_samples - self.stride_left_samples - self.stride_right_samples
        
        # Buffer to hold incoming audio
        self.buffer = np.array([], dtype=np.float32)
        
        # Timestamp tracking
        self.processed_samples = 0

    def add_audio(self, audio: np.ndarray):
        """Add audio to the buffer."""
        if audio.ndim > 1:
            # Convert to mono if needed (simple average)
            audio = audio.mean(axis=0)
        
        self.buffer = np.concatenate([self.buffer, audio])

    def available_chunks(self) -> Iterator[Tuple[np.ndarray, float]]:
        """
        Yield available chunks for processing.
        Returns: (chunk_audio, chunk_end_timestamp)
        """
        while len(self.buffer) >= self.chunk_samples:
            # Extract chunk
            chunk = self.buffer[:self.chunk_samples]
            
            # Calculate timestamp for the END of the KEPT part (middle part)
            # The kept part ends at stride_left + keep_duration
            # But in absolute time, it is processed_samples + stride_left + keep_duration
            # Wait, let's define timestamp as the end of the valid text.
            # The valid text corresponds to the middle part.
            # Middle part start: stride_left
            # Middle part end: stride_left + keep
            
            current_chunk_start_time = self.processed_samples / self.sample_rate
            middle_end_time = current_chunk_start_time + self.stride_left + (self.keep_samples / self.sample_rate)
            
            yield chunk, middle_end_time
            
            # Slide window
            # We advance by keep_samples
            step = self.keep_samples
            self.buffer = self.buffer[step:]
            self.processed_samples += step

    def finish(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Process remaining audio."""
        # For the last part, we might not have enough for a full chunk.
        # But we should process what we have.
        # Ideally we pad it? Or just process it as is?
        # The spec implies we just process remaining.
        if len(self.buffer) > 0:
            # Pad to chunk size if needed, or just process.
            # For simplicity, let's just yield what's left if it's significant?
            # Or maybe we should pad with silence to match chunk size to keep model happy?
            # Let's pad with zeros.
            if len(self.buffer) < self.chunk_samples:
                padding = np.zeros(self.chunk_samples - len(self.buffer), dtype=np.float32)
                chunk = np.concatenate([self.buffer, padding])
            else:
                chunk = self.buffer[:self.chunk_samples] # Should be covered by loop above but just in case
            
            current_chunk_start_time = self.processed_samples / self.sample_rate
            # For the last chunk, the valid part is everything after stride_left?
            # Or maybe we just treat it normally.
            middle_end_time = current_chunk_start_time + (len(self.buffer) / self.sample_rate)
            
            yield chunk, middle_end_time
            self.buffer = np.array([], dtype=np.float32)


class CTCAlignmentExtractor:
    def __init__(self, token_decoder):
        self.token_decoder = token_decoder
        # Wav2Vec2 downsampling factor is usually 320 (16kHz -> 50Hz)
        self.downsample_rate = 320 
        self.model_hz = 50

    def extract_middle(
        self, 
        logits: torch.Tensor, 
        stride_left_seconds: float, 
        stride_right_seconds: float
    ) -> str:
        """
        Extract text from the middle part of the logits.
        """
        # 1. Argmax
        pred_ids = torch.argmax(logits, dim=-1)
        
        # 2. Calculate frames to drop
        stride_left_frames = int(stride_left_seconds * self.model_hz)
        stride_right_frames = int(stride_right_seconds * self.model_hz)
        
        # 3. Slice (Keep middle)
        total_frames = pred_ids.size(0)
        
        # Ensure we don't slice out of bounds
        start_frame = min(stride_left_frames, total_frames)
        end_frame = max(start_frame, total_frames - stride_right_frames)
        
        middle_ids = pred_ids[start_frame:end_frame]
        
        if len(middle_ids) == 0:
            return ""

        # 4. Collapse (CTC) & Decode
        # Remove duplicates
        unique_ids = torch.unique_consecutive(middle_ids)
        # Remove blanks (assuming 0 is blank, which is standard for fairseq2/wav2vec2)
        # But we should check tokenizer or model config if possible. 
        # Usually pad_idx is 0 and it is used as blank in fairseq2.
        non_blank_ids = unique_ids[unique_ids != 0]
        
        text = self.token_decoder(non_blank_ids)
        return text


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
        self.device = torch.device(device) if torch.cuda.is_available() and device == "cuda" else torch.device("cpu")
        self.dtype = dtype
        self.chunk_duration = chunk_duration
        self.stride_left = stride_left
        self.stride_right = stride_right
        self.lang = lang  # Store language for LLM models
        
        # Initialize Base Pipeline to load model
        print(f"Loading model: {model_card}...")
        self.base_pipeline = ASRInferencePipeline(
            model_card=model_card,
            device=self.device,
            dtype=self.dtype
        )
        self.model = self.base_pipeline.model
        self.tokenizer = self.base_pipeline.tokenizer
        self.token_decoder = self.base_pipeline.token_decoder
        
        # Components
        self.buffer = StrideAudioBuffer(
            chunk_duration=chunk_duration,
            stride_left=stride_left,
            stride_right=stride_right
        )
        self.ctc_extractor = CTCAlignmentExtractor(self.token_decoder)
        
        print("Streaming pipeline initialized.")

    def add_audio(self, audio: np.ndarray):
        """Add audio data to the buffer."""
        self.buffer.add_audio(audio)

    def transcribe_available(self) -> Iterator[StreamingResult]:
        """Process all available chunks in the buffer."""
        for chunk, timestamp in self.buffer.available_chunks():
            text = self._process_chunk_optimized(chunk)
            if text:
                yield StreamingResult(
                    text=text,
                    is_final=False,
                    timestamp=timestamp,
                    latency=0.0 # TODO: Calculate latency
                )

    def finish(self) -> Iterator[StreamingResult]:
        """Process remaining audio."""
        for chunk, timestamp in self.buffer.finish():
            text = self._process_chunk_optimized(chunk)
            if text:
                yield StreamingResult(
                    text=text,
                    is_final=True,
                    timestamp=timestamp,
                    latency=0.0
                )

    def _process_chunk_optimized(self, audio_chunk: np.ndarray) -> str:
        # 1. Prepare Tensor
        waveform = torch.from_numpy(audio_chunk).float().to(self.device)
        
        # Handle dtype
        if self.dtype == torch.float16:
            waveform = waveform.half()
        elif self.dtype == torch.bfloat16:
            waveform = waveform.bfloat16()
        
        # 2. Normalize (Zero-mean, Unit-variance)
        # Using layer_norm to match training/inference pipeline
        with torch.no_grad():
             # layer_norm expects [..., dim], here we have [T]
             # We want to normalize across Time? 
             # The utils.py uses layer_norm(waveform, waveform.shape) which normalizes the whole tensor.
             # If waveform is [T], it normalizes across T.
             waveform = layer_norm(waveform, waveform.shape)

        # 3. Batch Layout
        # Model expects [Batch, Time]
        waveform = waveform.unsqueeze(0) # [1, T]
        seq_lens = torch.tensor([waveform.size(1)], device=self.device, dtype=torch.int64)
        
        batch_layout = BatchLayout(
            waveform.shape, 
            seq_lens=seq_lens,
            device=self.device
        )

        # 4. Model Forward
        with torch.inference_mode():
            # Wav2Vec2AsrModel returns (logits, padding_mask)
            # logits: [Batch, Time, Vocab]
            logits, _ = self.model(waveform, batch_layout)
        
        # 5. Extract Text
        return self.ctc_extractor.extract_middle(
            logits[0], 
            self.stride_left, 
            self.stride_right
        )
