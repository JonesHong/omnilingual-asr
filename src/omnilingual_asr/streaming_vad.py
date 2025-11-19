# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
import warnings

import numpy as np
import torch
from fairseq2.nn.batch_layout import BatchLayout
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from torch.nn.functional import layer_norm

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel


@dataclass
class StreamingResult:
    text: str          # The text recognized in this chunk
    is_final: bool     # Whether this result is final (usually for VAD segmentation)
    timestamp: float   # The end timestamp of this text in the stream
    latency: float     # Processing latency


class VADSegmenter:
    """Voice Activity Detection based audio segmentation."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,  # Reduced from 250ms
        min_silence_duration_ms: int = 500,  # Reduced from 500ms for faster response
        max_segment_duration_ms: int = 2000,  # Reduced from 2000ms
        speech_pad_ms: int = 30,
    ):
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_segment_duration_ms = max_segment_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        # Load Silero VAD
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.vad_model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load Silero VAD: {e}. VAD will be disabled.")
            self.vad_model = None
        
        # Buffer state
        self.buffer = np.array([], dtype=np.float32)
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_speech = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.processed_samples = 0
        
    def add_audio(self, audio: np.ndarray):
        """Add audio to the buffer."""
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        self.buffer = np.concatenate([self.buffer, audio])
    
    def get_speech_segments(self) -> Iterator[Tuple[np.ndarray, float, bool]]:
        """
        Yield speech segments.
        Returns: (audio_segment, end_timestamp, is_final)
        """
        if self.vad_model is None:
            # No VAD, just yield chunks
            chunk_size = int(3.0 * self.sample_rate)
            while len(self.buffer) >= chunk_size:
                chunk = self.buffer[:chunk_size]
                self.buffer = self.buffer[chunk_size:]
                timestamp = (self.processed_samples + chunk_size) / self.sample_rate
                self.processed_samples += chunk_size
                yield chunk, timestamp, False
            return
        
        # Process with VAD
        window_size = 512  # Silero VAD uses 512 samples
        
        while len(self.buffer) >= window_size:
            window = self.buffer[:window_size]
            
            # Get VAD probability
            with torch.no_grad():
                speech_prob = self.vad_model(
                    torch.from_numpy(window).float(),
                    self.sample_rate
                ).item()
            
            is_speech = speech_prob > 0.5
            
            if is_speech:
                self.speech_buffer = np.concatenate([self.speech_buffer, window])
                self.speech_frames += 1
                self.silence_frames = 0
                self.is_speech = True
                
                # Check if we've exceeded max segment duration
                max_segment_samples = int(self.max_segment_duration_ms * self.sample_rate / 1000)
                if len(self.speech_buffer) >= max_segment_samples:
                    # Force split even though still speaking
                    timestamp = (self.processed_samples + len(self.speech_buffer)) / self.sample_rate
                    yield self.speech_buffer.copy(), timestamp, False  # Not final, just forced split
                    
                    # IMPORTANT: Reset state to avoid duplicate processing
                    self.speech_buffer = np.array([], dtype=np.float32)
                    self.speech_frames = 0
                    self.is_speech = False  # Reset speech state
                    self.silence_frames = 0
            else:
                self.silence_frames += 1
                
                # If we were in speech and now have enough silence
                min_silence_frames = int(self.min_silence_duration_ms * self.sample_rate / 1000 / window_size)
                min_speech_frames = int(self.min_speech_duration_ms * self.sample_rate / 1000 / window_size)
                
                if self.is_speech and self.silence_frames >= min_silence_frames:
                    # End of speech segment
                    if self.speech_frames >= min_speech_frames and len(self.speech_buffer) > 0:
                        timestamp = (self.processed_samples + len(self.speech_buffer)) / self.sample_rate
                        yield self.speech_buffer.copy(), timestamp, True
                        self.speech_buffer = np.array([], dtype=np.float32)
                    
                    self.is_speech = False
                    self.speech_frames = 0
            
            self.buffer = self.buffer[window_size:]
            self.processed_samples += window_size
    
    def finish(self) -> Iterator[Tuple[np.ndarray, float, bool]]:
        """Process remaining audio."""
        if len(self.speech_buffer) > 0:
            timestamp = (self.processed_samples + len(self.speech_buffer)) / self.sample_rate
            yield self.speech_buffer.copy(), timestamp, True
            self.speech_buffer = np.array([], dtype=np.float32)
        
        if len(self.buffer) > 0:
            timestamp = (self.processed_samples + len(self.buffer)) / self.sample_rate
            yield self.buffer.copy(), timestamp, True
            self.buffer = np.array([], dtype=np.float32)


class StreamingASRPipelineVAD:
    """VAD-based streaming ASR pipeline supporting both CTC and LLM models."""
    
    def __init__(
        self,
        model_card: str = "omniASR_CTC_1B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lang: Optional[str] = None,
        use_vad: bool = True,
        min_speech_duration_ms: int = 250,  # Reduced for faster response
        min_silence_duration_ms: int = 500,  # Reduced for faster response
        max_segment_duration_ms: int = 2000,  # Reduced for faster response
    ):
        self.device = torch.device(device) if torch.cuda.is_available() and device == "cuda" else torch.device("cpu")
        self.dtype = dtype
        self.lang = lang
        
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
        
        # Determine model type
        self.is_ctc = isinstance(self.model, Wav2Vec2AsrModel)
        self.is_llm = isinstance(self.model, Wav2Vec2LlamaModel)
        
        print(f"Model type: {'CTC' if self.is_ctc else 'LLM'}")
        
        # VAD Segmenter
        if use_vad:
            self.segmenter = VADSegmenter(
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                max_segment_duration_ms=max_segment_duration_ms
            )
        else:
            self.segmenter = VADSegmenter()
            self.segmenter.vad_model = None  # Disable VAD
        
        print("VAD-based streaming pipeline initialized.")

    def add_audio(self, audio: np.ndarray):
        """Add audio data to the buffer."""
        self.segmenter.add_audio(audio)

    def transcribe_available(self) -> Iterator[StreamingResult]:
        """Process all available speech segments."""
        for segment, timestamp, is_final in self.segmenter.get_speech_segments():
            if len(segment) < 1600:  # Skip very short segments (< 0.1s)
                continue
            
            text = self._process_segment(segment)
            if text:
                yield StreamingResult(
                    text=text,
                    is_final=is_final,
                    timestamp=timestamp,
                    latency=0.0
                )

    def finish(self) -> Iterator[StreamingResult]:
        """Process remaining audio."""
        for segment, timestamp, is_final in self.segmenter.finish():
            if len(segment) < 1600:
                continue
            
            text = self._process_segment(segment)
            if text:
                yield StreamingResult(
                    text=text,
                    is_final=True,
                    timestamp=timestamp,
                    latency=0.0
                )

    def _process_segment(self, audio_segment: np.ndarray) -> str:
        """Process a complete audio segment."""
        # Prepare input
        inp = [{
            "waveform": torch.from_numpy(audio_segment),
            "sample_rate": 16000,
        }]
        
        # Use base pipeline's transcribe method
        if self.is_llm and self.lang:
            result = self.base_pipeline.transcribe(
                inp=inp,
                batch_size=1,
                lang=[self.lang]
            )
        else:
            result = self.base_pipeline.transcribe(
                inp=inp,
                batch_size=1,
                lang=[self.lang] if self.lang else None
            )
        
        return result[0] if result else ""
