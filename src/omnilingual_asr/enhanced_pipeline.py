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
    JSON = "json"  # [{"text": "文字", "start": 0.0, "end": 5.0}]


class TimeFormat(Enum):
    """時間格式"""
    AUTO = "auto"        # 根據總長度自動選擇
    SECONDS = "seconds"  # 5.2s
    MMSS = "mm:ss"       # 00:05
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
        base_pipeline: Optional[ASRInferencePipeline] = None,
    ):
        """
        初始化增強版 ASR Pipeline
        
        Args:
            model_card: 模型名稱
            device: 運算裝置
            dtype: 數據類型
            base_pipeline: 可選的已載入 pipeline，用於共享模型
        """
        if base_pipeline is not None:
            self.base_pipeline = base_pipeline
        else:
            self.base_pipeline = ASRInferencePipeline(
                model_card=model_card,
                device=device,
                dtype=dtype
            )
        self.model_card = model_card
        self.device = device
        self.dtype = dtype
        
        # Initialize VAD for smart splitting
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.vad_model.to(torch.device("cpu")) # VAD is fast enough on CPU
            self.vad_model.eval()
            print("VAD model loaded for smart chunking.")
        except Exception as e:
            print(f"Warning: Failed to load VAD model ({e}). Fallback to fixed chunking.")
            self.vad_model = None
    
    def transcribe(
        self,
        inp: List[Dict[str, Any]],
        batch_size: int = 1,
        lang: Optional[List[str]] = None,
        # 切割參數
        chunk_duration: float = 30.0,
        overlap: float = 1.0,
        chunk_separator: str = " ",  # 新增：chunk 之間的分隔符
        # VAD 切割參數
        split_on_silence: bool = False,  # 是否強制使用 VAD 依靜音切割
        min_silence_duration_ms: int = 500, # 最小靜音長度 (毫秒)
        speech_pad_ms: int = 100, # 語音前後保留的緩衝 (毫秒)
        # 時間戳記參數
        timestamp_format: TimestampFormat | str = TimestampFormat.NONE,
        time_format: TimeFormat | str = TimeFormat.AUTO,  # Default to AUTO
        timestamp_template: Optional[str] = None,
    ) -> List[str | List[Dict[str, Any]]]:
        """
        轉譯音訊，自動處理長音訊切割
        
        Args:
            inp: 輸入音訊列表，格式 [{"waveform": tensor, "sample_rate": int}]
            batch_size: 批次大小
            lang: 語言代碼列表
            chunk_duration: 每段音訊長度（秒），必須 <= 40
            overlap: 重疊長度（秒），避免句子被切斷
            chunk_separator: chunk 之間的分隔符（預設為空格，可設為 "" 或 "\n" 等）
            split_on_silence: 是否強制使用 VAD 依靜音切割 (忽略 chunk_duration 限制)
            min_silence_duration_ms: 認定為靜音的最小長度 (毫秒)
            speech_pad_ms: 語音前後保留的緩衝 (毫秒)
            timestamp_format: 時間戳記格式 (none/simple/detailed/json)
            time_format: 時間顯示格式 (auto/seconds/mm:ss/hh:mm:ss)
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
            
            # Determine time format based on duration if AUTO is selected
            effective_time_format = time_format
            if time_format == TimeFormat.AUTO:
                if audio_duration < 60:
                    effective_time_format = TimeFormat.SECONDS
                elif audio_duration < 3600:
                    effective_time_format = TimeFormat.MMSS
                else:
                    effective_time_format = TimeFormat.HHMMSS
            
            # 判斷是否需要切割
            if split_on_silence and self.vad_model is not None:
                # 使用 VAD 強制切割
                chunks = self._split_by_silence(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_pad_ms=speech_pad_ms
                )
                
                # 逐段處理 (Batch Processing)
                # 為了效率，我們可以將所有 chunks 收集起來一次 batch 處理
                # 但這裡為了保持程式碼結構簡單，且 base_pipeline.transcribe 本身支援 batch，
                # 我們可以將 chunks 組合成 batch 輸入。
                
                chunk_inputs = []
                for _, chunk_waveform in chunks:
                    chunk_inputs.append({
                        "waveform": chunk_waveform,
                        "sample_rate": sample_rate
                    })
                
                # 批次轉譯
                # 注意：如果 chunks 數量很大，可能需要分批送入，避免 OOM
                # 這裡簡單起見，直接依賴 base_pipeline 的 batch_size 參數處理可能不夠，
                # 因為 base_pipeline.transcribe 的 batch_size 是指 inference 的 batch size。
                # 我們需要自己手動分批送入 base_pipeline.transcribe
                
                chunk_texts = []
                
                for i in range(0, len(chunk_inputs), batch_size):
                    batch_inp = chunk_inputs[i : i + batch_size]
                    batch_results = self.base_pipeline.transcribe(
                        inp=batch_inp,
                        batch_size=len(batch_inp), # 這裡已經手動 batch 了
                        lang=lang
                    )
                    chunk_texts.extend(batch_results)
                
                # 組合結果
                chunk_results = []
                for i, (start_time, chunk_waveform) in enumerate(chunks):
                    chunk_duration_actual = len(chunk_waveform) / sample_rate
                    end_time = start_time + chunk_duration_actual
                    
                    chunk_results.append(ChunkResult(
                        text=chunk_texts[i],
                        start_time=start_time,
                        end_time=end_time,
                        duration=chunk_duration_actual
                    ))
                
                combined = self._combine_chunks(
                    chunk_results,
                    timestamp_format,
                    effective_time_format,
                    timestamp_template,
                    chunk_separator
                )
                results.append(combined)

            elif audio_duration <= self.MAX_AUDIO_DURATION:
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
                        time_format=effective_time_format,
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
                
                # 組合分塊結果
                combined = self._combine_chunks(
                    chunk_results,
                    timestamp_format,
                    effective_time_format,
                    timestamp_template,
                    chunk_separator
                )
                results.append(combined)
        
        return results
    
    def _find_split_point(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_sample: int,
        window_sample: int
    ) -> int:
        """
        Find the best split point near target_sample within window_sample.
        Returns the sample index to split at.
        """
        if self.vad_model is None:
            return target_sample
            
        # Define search range: [target - window, target + window]
        start_search = max(0, target_sample - window_sample)
        end_search = min(len(waveform), target_sample + window_sample)
        
        if start_search >= end_search:
            return target_sample
            
        search_region = waveform[start_search:end_search]
        
        # Get speech probabilities
        # Silero expects (batch, samples)
        with torch.no_grad():
            # Process in chunks of 512 samples (32ms)
            vad_step = 512
            probs = []
            for i in range(0, len(search_region), vad_step):
                chunk = search_region[i:i+vad_step]
                if len(chunk) < vad_step:
                    break
                prob = self.vad_model(chunk.unsqueeze(0), sample_rate).item()
                probs.append(prob)
        
        # Find the longest sequence of low probability (silence)
        # We map probs back to sample indices
        # This is a simplified heuristic: find the index with minimum probability
        # or the center of the longest silence run.
        
        if not probs:
            return target_sample
            
        # Simple heuristic: find the window with lowest average speech probability
        # We want to cut where it is most silent.
        
        min_prob = 1.0
        best_idx = -1
        
        # Check every 32ms step
        for i, prob in enumerate(probs):
            if prob < min_prob:
                min_prob = prob
                best_idx = i
        
        if best_idx != -1:
            # Convert back to absolute sample index
            # best_idx is the index of the 512-sample chunk
            relative_sample = best_idx * 512
            return start_search + relative_sample
            
        return target_sample

    def _split_by_silence(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        使用 VAD 將音訊依照靜音切割成多個片段
        
        Args:
            waveform: 音訊波形
            sample_rate: 採樣率
            min_silence_duration_ms: 最小靜音長度
            speech_pad_ms: 語音前後保留的緩衝
            
        Returns:
            [(start_time, chunk_waveform), ...]
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
            
        if self.vad_model is None:
            print("Warning: VAD model not loaded. Returning original waveform.")
            return [(0.0, waveform)]

        # 使用 silero-vad 的 get_speech_timestamps
        # 這裡我們需要手動呼叫 utils，因為我們是直接載入模型
        # 為了避免依賴額外的 utils 文件，我們實作一個簡化的版本或使用模型提供的功能
        
        # Silero VAD 官方推薦使用 get_speech_timestamps
        # 嘗試從 torch.hub 載入 utils
        try:
            (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )[1:] # model is the first return, we need the rest
             # Wait, torch.hub.load returns the model if 'model' arg is specified.
             # To get utils, we usually load the repo.
             # Let's try a safer way: manually implement a simple thresholding loop or use the model directly.
             # Actually, the user environment might not have internet access to reload hub.
             # Let's assume we can use the model we already loaded.
            pass
        except:
            pass

        # 由於環境限制，我們手動實作一個基於機率的切割
        # Process in chunks
        window_size_samples = 512
        
        speech_probs = []
        with torch.no_grad():
            for i in range(0, len(waveform), window_size_samples):
                chunk = waveform[i:i+window_size_samples]
                if len(chunk) < window_size_samples:
                    # Pad last chunk
                    pad_size = window_size_samples - len(chunk)
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
                prob = self.vad_model(chunk.unsqueeze(0), sample_rate).item()
                speech_probs.append(prob)
        
        # Convert probs to timestamps
        threshold = 0.5
        is_speech = False
        speech_segments = []
        current_start = 0
        
        # 簡單的狀態機
        # 為了符合 min_silence_duration_ms，我們需要更複雜的邏輯
        # 這裡簡化處理：只要機率 > 0.5 視為語音
        # 然後合併距離很近的語音段
        
        raw_segments = []
        start_idx = -1
        
        for i, prob in enumerate(speech_probs):
            if prob >= threshold:
                if start_idx == -1:
                    start_idx = i
            else:
                if start_idx != -1:
                    end_idx = i
                    raw_segments.append((start_idx, end_idx))
                    start_idx = -1
        
        if start_idx != -1:
            raw_segments.append((start_idx, len(speech_probs)))
            
        # 合併過短的靜音 (Merge segments that are close together)
        # min_silence_duration_ms 轉換為 window 數量
        # window_duration_ms = 512 / sample_rate * 1000
        window_duration_ms = (window_size_samples / sample_rate) * 1000
        min_silence_windows = int(min_silence_duration_ms / window_duration_ms)
        
        if not raw_segments:
            return []
            
        merged_segments = []
        current_seg = raw_segments[0]
        
        for next_seg in raw_segments[1:]:
            silence_gap = next_seg[0] - current_seg[1]
            if silence_gap < min_silence_windows:
                # Merge
                current_seg = (current_seg[0], next_seg[1])
            else:
                merged_segments.append(current_seg)
                current_seg = next_seg
        merged_segments.append(current_seg)
        
        # 轉換回時間並加上 padding
        final_chunks = []
        pad_windows = int(speech_pad_ms / window_duration_ms)
        
        for start_win, end_win in merged_segments:
            start_win = max(0, start_win - pad_windows)
            end_win = min(len(speech_probs), end_win + pad_windows)
            
            start_sample = start_win * window_size_samples
            end_sample = end_win * window_size_samples
            
            # Ensure valid range
            end_sample = min(end_sample, len(waveform))
            
            chunk_waveform = waveform[start_sample:end_sample]
            start_time = start_sample / sample_rate
            
            final_chunks.append((start_time, chunk_waveform))
            
        return final_chunks

    def _split_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_duration: float,
        overlap: float
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        切割音訊為多個片段，使用 VAD 尋找最佳切割點
        
        Args:
            waveform: 音訊波形
            sample_rate: 採樣率
            chunk_duration: 目標每段長度（秒）
            overlap: 最小重疊長度（秒）
            
        Returns:
            [(start_time, chunk_waveform), ...]
        """
        # 確保是 1D tensor
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        total_samples = len(waveform)
        target_chunk_samples = int(chunk_duration * sample_rate)
        min_overlap_samples = int(overlap * sample_rate)
        search_window_samples = int(3.0 * sample_rate) # +/- 3 seconds
        
        chunks = []
        start_sample = 0
        
        while start_sample < total_samples:
            # Calculate ideal end point
            ideal_end = start_sample + target_chunk_samples
            
            if ideal_end >= total_samples:
                # Last chunk
                end_sample = total_samples
                next_start = total_samples # Loop will terminate
            else:
                # Find best split point near ideal_end
                split_sample = self._find_split_point(
                    waveform, 
                    sample_rate, 
                    ideal_end, 
                    search_window_samples
                )
                
                end_sample = split_sample
                
                # Next chunk starts before current chunk ends (overlap)
                # But we want to respect the split point we found.
                # If we split at a silence, the next chunk should ideally start 
                # a bit before that silence to provide context, OR 
                # if we treat it as a clean cut, we might not need much overlap.
                # However, to be safe and consistent with the API, we ensure 
                # the next chunk starts 'overlap' seconds before the split point.
                
                next_start = max(start_sample + 1000, end_sample - min_overlap_samples)
            
            chunk = waveform[start_sample:end_sample]
            start_time = start_sample / sample_rate
            chunks.append((start_time, chunk))
            
            start_sample = next_start
        
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
    ) -> str | Dict[str, Any]:
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
            格式化後的字串或字典
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
        elif timestamp_format == TimestampFormat.JSON:
            return {
                "text": text,
                "start": start_time,
                "end": end_time
            }
        else:
            return text
    
    def _combine_chunks(
        self,
        chunks: List[ChunkResult],
        timestamp_format: TimestampFormat,
        time_format: TimeFormat,
        template: Optional[str] = None,
        separator: str = " "
    ) -> str | List[Dict[str, Any]]:
        """
        合併多個片段的結果
        
        Args:
            chunks: 片段結果列表
            timestamp_format: 時間戳記格式
            time_format: 時間顯示格式
            template: 自訂模板
            separator: chunk 之間的分隔符
            
        Returns:
            合併後的字串或字典列表
        """
        if timestamp_format == TimestampFormat.NONE:
            # 無時間戳記，直接合併文字
            return separator.join(chunk.text for chunk in chunks)
        else:
            # 有時間戳記
            if timestamp_format == TimestampFormat.JSON:
                # JSON 格式直接返回列表
                return [
                    self._format_result(
                        text=chunk.text,
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        timestamp_format=timestamp_format,
                        time_format=time_format,
                        template=template
                    )
                    for chunk in chunks
                ]
            
            # 其他格式，每段一行
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
