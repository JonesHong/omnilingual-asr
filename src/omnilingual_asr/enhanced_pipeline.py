# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced ASR Pipeline with auto-chunking and timestamp support.
Wrapper around ASRInferencePipeline without modifying the core class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import torch

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


class TimestampFormat(Enum):
    """時間戳記顯示格式"""
    NONE = "none"  # 無時間戳記
    SIMPLE = "simple"  # [00:05] 文字
    DETAILED = "detailed"  # [00:05 - 00:08] 文字


class TimeFormat(Enum):
    """時間格式"""
    SECONDS = "seconds"  # 5.2s
    MMSS = "mm:ss"  # 00:05
    HHMMSS = "hh:mm:ss"  # 00:00:05


@dataclass
class ChunkResult:
    """單個音訊片段的辨識結果"""
    text: str
    start_time: float
    end_time: float
    duration: float


class EnhancedASRPipeline:
    """
    增強版 ASR Pipeline，支援：
    1. 自動音訊切割（超過模型限制時）
    2. 可配置的時間戳記顯示
    3. 靈活的時間格式
    
    不修改核心 ASRInferencePipeline，僅作為包裝層。
    """
    
    # 模型限制（不可配置）
    MAX_AUDIO_DURATION = 40.0  # 秒
    
    def __init__(
        self,
        model_card: str = "omniASR_CTC_1B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        初始化增強版 ASR Pipeline
        
        Args:
            model_card: 模型名稱
            device: 運算裝置
            dtype: 數據類型
        """
        self.base_pipeline = ASRInferencePipeline(
            model_card=model_card,
            device=device,
            dtype=dtype
        )
        self.model_card = model_card
        self.device = device
        self.dtype = dtype
    
    def transcribe(
        self,
        inp: List[Dict[str, Any]],
        batch_size: int = 1,
        lang: Optional[List[str]] = None,
        # 切割參數
        chunk_duration: float = 30.0,
        overlap: float = 1.0,
        # 時間戳記參數
        timestamp_format: TimestampFormat | str = TimestampFormat.NONE,
        time_format: TimeFormat | str = TimeFormat.MMSS,
        timestamp_template: Optional[str] = None,
    ) -> List[str]:
        """
        轉譯音訊，自動處理長音訊切割
        
        Args:
            inp: 輸入音訊列表，格式 [{"waveform": tensor, "sample_rate": int}]
            batch_size: 批次大小
            lang: 語言代碼列表
            chunk_duration: 每段音訊長度（秒），必須 <= 40
            overlap: 重疊長度（秒），避免句子被切斷
            timestamp_format: 時間戳記格式 (none/simple/detailed)
            time_format: 時間顯示格式 (seconds/mm:ss/hh:mm:ss)
            timestamp_template: 自訂時間戳記模板，例如 "[{start}] {text}"
            
        Returns:
            轉譯結果列表
            
        Raises:
            ValueError: 如果 chunk_duration > 40
        """
        # 驗證參數
        if chunk_duration > self.MAX_AUDIO_DURATION:
            raise ValueError(
                f"chunk_duration ({chunk_duration}s) 不能超過模型限制 "
                f"({self.MAX_AUDIO_DURATION}s)"
            )
        
        # 轉換枚舉
        if isinstance(timestamp_format, str):
            timestamp_format = TimestampFormat(timestamp_format)
        if isinstance(time_format, str):
            time_format = TimeFormat(time_format)
        
        results = []
        
        for audio_dict in inp:
            waveform = audio_dict["waveform"]
            sample_rate = audio_dict["sample_rate"]
            
            # 計算音訊長度
            if waveform.dim() == 1:
                audio_duration = len(waveform) / sample_rate
            else:
                audio_duration = waveform.shape[-1] / sample_rate
            
            # 判斷是否需要切割
            if audio_duration <= self.MAX_AUDIO_DURATION:
                # 不需要切割，直接處理
                result = self.base_pipeline.transcribe(
                    inp=[audio_dict],
                    batch_size=batch_size,
                    lang=lang
                )
                
                # 格式化輸出
                if timestamp_format == TimestampFormat.NONE:
                    results.append(result[0])
                else:
                    formatted = self._format_result(
                        text=result[0],
                        start_time=0.0,
                        end_time=audio_duration,
                        timestamp_format=timestamp_format,
                        time_format=time_format,
                        template=timestamp_template
                    )
                    results.append(formatted)
            else:
                # 需要切割
                chunks = self._split_audio(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    chunk_duration=chunk_duration,
                    overlap=overlap
                )
                
                # 逐段處理
                chunk_results = []
                for start_time, chunk_waveform in chunks:
                    chunk_dict = {
                        "waveform": chunk_waveform,
                        "sample_rate": sample_rate
                    }
                    
                    result = self.base_pipeline.transcribe(
                        inp=[chunk_dict],
                        batch_size=batch_size,
                        lang=lang
                    )
                    
                    chunk_duration_actual = len(chunk_waveform) / sample_rate
                    end_time = start_time + chunk_duration_actual
                    
                    chunk_results.append(ChunkResult(
                        text=result[0],
                        start_time=start_time,
                        end_time=end_time,
                        duration=chunk_duration_actual
                    ))
                
                # 合併結果
                combined = self._combine_chunks(
                    chunks=chunk_results,
                    timestamp_format=timestamp_format,
                    time_format=time_format,
                    template=timestamp_template
                )
                results.append(combined)
        
        return results
    
    def _split_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_duration: float,
        overlap: float
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        切割音訊為多個片段
        
        Args:
            waveform: 音訊波形
            sample_rate: 採樣率
            chunk_duration: 每段長度（秒）
            overlap: 重疊長度（秒）
            
        Returns:
            [(start_time, chunk_waveform), ...]
        """
        # 確保是 1D tensor
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        total_samples = len(waveform)
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start_sample = 0
        
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[start_sample:end_sample]
            
            start_time = start_sample / sample_rate
            chunks.append((start_time, chunk))
            
            if end_sample >= total_samples:
                break
            
            start_sample += step_samples
        
        return chunks
    
    def _format_time(self, seconds: float, time_format: TimeFormat) -> str:
        """
        格式化時間
        
        Args:
            seconds: 秒數
            time_format: 時間格式
            
        Returns:
            格式化後的時間字串
        """
        if time_format == TimeFormat.SECONDS:
            return f"{seconds:.1f}s"
        elif time_format == TimeFormat.MMSS:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        elif time_format == TimeFormat.HHMMSS:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{seconds:.1f}s"
    
    def _format_result(
        self,
        text: str,
        start_time: float,
        end_time: float,
        timestamp_format: TimestampFormat,
        time_format: TimeFormat,
        template: Optional[str] = None
    ) -> str:
        """
        格式化單個結果
        
        Args:
            text: 文字內容
            start_time: 開始時間
            end_time: 結束時間
            timestamp_format: 時間戳記格式
            time_format: 時間顯示格式
            template: 自訂模板
            
        Returns:
            格式化後的字串
        """
        if timestamp_format == TimestampFormat.NONE:
            return text
        
        start_str = self._format_time(start_time, time_format)
        end_str = self._format_time(end_time, time_format)
        
        if template:
            # 使用自訂模板
            return template.format(
                start=start_str,
                end=end_str,
                text=text,
                duration=self._format_time(end_time - start_time, time_format)
            )
        elif timestamp_format == TimestampFormat.SIMPLE:
            return f"[{start_str}] {text}"
        elif timestamp_format == TimestampFormat.DETAILED:
            return f"[{start_str} - {end_str}] {text}"
        else:
            return text
    
    def _combine_chunks(
        self,
        chunks: List[ChunkResult],
        timestamp_format: TimestampFormat,
        time_format: TimeFormat,
        template: Optional[str] = None
    ) -> str:
        """
        合併多個片段的結果
        
        Args:
            chunks: 片段結果列表
            timestamp_format: 時間戳記格式
            time_format: 時間顯示格式
            template: 自訂模板
            
        Returns:
            合併後的字串
        """
        if timestamp_format == TimestampFormat.NONE:
            # 無時間戳記，直接合併文字
            return " ".join(chunk.text for chunk in chunks)
        else:
            # 有時間戳記，每段一行
            lines = []
            for chunk in chunks:
                formatted = self._format_result(
                    text=chunk.text,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    timestamp_format=timestamp_format,
                    time_format=time_format,
                    template=template
                )
                lines.append(formatted)
            return "\n".join(lines)
