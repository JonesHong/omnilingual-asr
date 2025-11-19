#!/usr/bin/env python3
"""
使用真實音訊檔案測試 Omnilingual ASR（支援長音訊分段處理）
測試檔案：什麼是上帝的道.mp3
"""

import torch
import torchaudio
import time
from pathlib import Path
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def load_audio(audio_path: str, target_sr: int = 16000):
    """載入音訊檔案並重採樣到目標採樣率"""
    print(f"\n載入音訊檔案: {audio_path}")

    # 載入音訊
    waveform, sample_rate = torchaudio.load(audio_path)

    # 如果是立體聲，轉為單聲道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print(f"  已轉換為單聲道")

    # 重採樣到目標採樣率
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        print(f"  已重採樣: {sample_rate} Hz -> {target_sr} Hz")

    # 移除批次維度
    waveform = waveform.squeeze(0)

    duration = len(waveform) / target_sr
    print(f"  音訊長度: {duration:.2f} 秒")
    print(f"  音訊形狀: {waveform.shape}")

    return waveform, target_sr


def split_audio(waveform, sample_rate: int, chunk_duration: float = 30.0, overlap: float = 1.0):
    """
    將長音訊分段處理

    Args:
        waveform: 音訊波形
        sample_rate: 採樣率
        chunk_duration: 每段長度（秒）
        overlap: 重疊長度（秒），避免句子被切斷

    Returns:
        chunks: 分段列表，每個元素包含 (start_time, waveform_chunk)
    """
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


def test_asr(
    audio_path: str,
    model_card: str = "omniASR_CTC_300M",
    device: str = "cuda",
    lang: str = "cmn_Hant",  # 中文
    chunk_duration: float = 30.0,  # 每段 30 秒
    overlap: float = 1.0,  # 重疊 1 秒
):
    """測試 ASR 系統（支援長音訊）"""

    print("=" * 70)
    print("Omnilingual ASR 長音訊測試")
    print("=" * 70)

    # 檢查裝置
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  GPU 不可用，切換到 CPU 模式")
        device = "cpu"

    if device == "cuda":
        print(f"\n✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"\n⚠️  使用 CPU 模式")

    # 載入音訊
    waveform, sample_rate = load_audio(audio_path)
    audio_duration = len(waveform) / sample_rate

    # 檢查是否需要分段
    MAX_DURATION = 40.0  # 模型限制
    if audio_duration > MAX_DURATION:
        print(f"\n⚠️  音訊長度 ({audio_duration:.2f}s) 超過限制 ({MAX_DURATION}s)")
        print(f"  將分段處理：每段 {chunk_duration}s，重疊 {overlap}s")
        chunks = split_audio(waveform, sample_rate, chunk_duration, overlap)
        print(f"  共分為 {len(chunks)} 段")
    else:
        chunks = [(0, waveform)]
        print(f"\n✓ 音訊長度在限制內，無需分段")

    # 載入模型
    print(f"\n載入模型: {model_card}")
    print(f"  目標語言: {lang} (中文)")

    start_time = time.time()
    pipeline = ASRInferencePipeline(
        model_card=model_card,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    load_time = time.time() - start_time
    print(f"✓ 模型載入完成 ({load_time:.2f}s)")

    # 執行辨識
    print("\n" + "=" * 70)
    print("開始語音辨識...")
    print("=" * 70)

    all_results = []
    total_inference_time = 0

    for i, (start_time_sec, chunk) in enumerate(chunks, 1):
        print(f"\n處理第 {i}/{len(chunks)} 段 (開始於 {start_time_sec:.1f}s)...")

        chunk_start = time.time()

        # 準備輸入
        inp = [{
            "waveform": chunk,
            "sample_rate": sample_rate,
        }]

        # 執行轉錄
        result = pipeline.transcribe(
            inp=inp,
            batch_size=1,
            lang=[lang],
        )

        chunk_time = time.time() - chunk_start
        total_inference_time += chunk_time

        chunk_duration_actual = len(chunk) / sample_rate
        rtf = chunk_time / chunk_duration_actual

        print(f"  ✓ 完成 ({chunk_time:.2f}s, RTF: {rtf:.3f})")
        print(f"  文字: {result[0][:80]}...")

        all_results.append({
            "start_time": start_time_sec,
            "duration": chunk_duration_actual,
            "text": result[0],
            "inference_time": chunk_time,
        })

    # 合併結果
    full_transcription = "\n".join([r["text"] for r in all_results])

    # 顯示結果
    print("\n" + "=" * 70)
    print("完整辨識結果")
    print("=" * 70)
    print(f"\n{full_transcription}\n")

    # 效能指標
    print("=" * 70)
    print("效能指標")
    print("=" * 70)
    print(f"  音訊總長度: {audio_duration:.2f} 秒")
    print(f"  分段數量: {len(chunks)}")
    print(f"  總處理時間: {total_inference_time:.2f} 秒")
    print(f"  平均 RTF: {total_inference_time / audio_duration:.3f}")
    print(f"  平均速度: {audio_duration / total_inference_time:.2f}x 實時速度")

    if device == "cuda":
        print(f"\nGPU 記憶體使用:")
        print(f"  已分配: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  最大使用: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

    # 字數統計
    total_chars = len(full_transcription)
    total_words = len(full_transcription.split())
    print(f"\n文字統計:")
    print(f"  總字數: {total_chars}")
    print(f"  總詞數: {total_words}")

    print("\n" + "=" * 70)
    print("✓ 測試完成！")
    print("=" * 70)

    return full_transcription, all_results


def main():
    """主程式"""

    # 音訊檔案路徑
    audio_path = "/mnt/c/work/omnilingual-asr/什麼是上帝的道.mp3"

    # 檢查檔案是否存在
    if not Path(audio_path).exists():
        print(f"❌ 錯誤：找不到音訊檔案: {audio_path}")
        return

    # 測試參數
    config = {
        "audio_path": audio_path,
        "model_card": "omniASR_LLM_3B",  # CTC 模型: omniASR_CTC_300M, omniASR_CTC_1B, omniASR_CTC_3B, omniASR_CTC_7B
                                          # LLM 模型: omniASR_LLM_300M, omniASR_LLM_1B, omniASR_LLM_3B, omniASR_LLM_7B
                                          # ⚠️ W2V 模型 (omniASR_W2V_*) 無法用於 ASR,只用於 SSL
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lang": "cmn",  # 中文（繁體/簡體通用）
        "chunk_duration": 30.0,  # 每段 30 秒
        "overlap": 1.0,  # 重疊 1 秒
    }

    print("\n測試配置:")
    print(f"  音訊檔案: {Path(audio_path).name}")
    print(f"  模型: {config['model_card']}")
    print(f"  裝置: {config['device']}")
    print(f"  語言: {config['lang']}")
    print(f"  分段長度: {config['chunk_duration']}s")
    print(f"  重疊長度: {config['overlap']}s")

    try:
        # 執行測試
        full_text, segment_results = test_asr(**config)

        # 儲存完整結果
        output_file = "transcription_result.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"音訊檔案: {Path(audio_path).name}\n")
            f.write(f"模型: {config['model_card']}\n")
            f.write(f"語言: {config['lang']}\n")
            f.write(f"分段數量: {len(segment_results)}\n")
            f.write(f"\n{'=' * 70}\n")
            f.write(f"完整辨識結果:\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(full_text)
            f.write(f"\n\n{'=' * 70}\n")
            f.write(f"分段詳細結果:\n")
            f.write(f"{'=' * 70}\n\n")

            for i, seg in enumerate(segment_results, 1):
                f.write(f"段落 {i} (開始於 {seg['start_time']:.1f}s):\n")
                f.write(f"{'-' * 70}\n")
                f.write(f"{seg['text']}\n\n")

        print(f"\n💾 辨識結果已儲存到: {output_file}")

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
