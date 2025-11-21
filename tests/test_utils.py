"""
測試工具模組 - 提供共用的測試輔助功能

包含：
- Tee: 同時輸出到多個文件的類
- TestResultSaver: 自動保存測試結果的上下文管理器
- load_audio: 載入並重採樣音訊的通用函數
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import torchaudio


class Tee:
    """同時輸出到多個文件"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


class TestResultSaver:
    """
    自動保存測試結果的上下文管理器
    
    使用方式:
        with TestResultSaver("test_name") as saver:
            # 執行測試
            print("測試輸出...")
    """
    
    def __init__(self, test_name: str, results_dir: Optional[Path] = None):
        """
        Args:
            test_name: 測試名稱（不含 .txt 副檔名）
            results_dir: 結果目錄，預設為 tests/results/
        """
        self.test_name = test_name
        
        if results_dir is None:
            # 假設從 tests/ 目錄下的腳本調用
            self.results_dir = Path(__file__).parent / "results"
        else:
            self.results_dir = results_dir
            
        self.results_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamped_file = self.results_dir / f"{test_name}_{timestamp}.txt"
        self.latest_file = self.results_dir / f"{test_name}_latest.txt"
        
        self.original_stdout = None
        self.file_handles = []
    
    def __enter__(self):
        """進入上下文，開始重定向輸出"""
        # 打開文件
        f_time = open(self.timestamped_file, 'w', encoding='utf-8')
        f_latest = open(self.latest_file, 'w', encoding='utf-8')
        self.file_handles = [f_time, f_latest]
        
        # 重定向 stdout
        self.original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f_time, f_latest)
        
        # 打印測試開始信息
        print(f"測試開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢復輸出並關閉文件"""
        # 打印測試結束信息
        print()
        print("=" * 70)
        print(f"測試結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 恢復 stdout
        if self.original_stdout:
            sys.stdout = self.original_stdout
        
        # 關閉文件
        for f in self.file_handles:
            f.close()
        
        # 打印保存信息
        print(f"\n✓ 結果已保存:")
        print(f"  - 時間戳版本: {self.timestamped_file}")
        print(f"  - 最新版本: {self.latest_file}")
        
        # 不抑制異常
        return False


def load_audio(audio_path: str, target_sr: int = 16000):
    """
    載入音訊並重採樣到目標採樣率
    
    Args:
        audio_path: 音訊文件路徑
        target_sr: 目標採樣率，預設 16000Hz
        
    Returns:
        (audio_numpy, sample_rate): 音訊數據（numpy array）和採樣率
    """
    print(f"Loading audio: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 轉為單聲道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 重採樣
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0).numpy(), target_sr


def get_test_audio_path(filename: str = "什麼是上帝的道.mp3") -> Path:
    """
    獲取測試音訊文件的路徑
    
    Args:
        filename: 音訊文件名，預設為專案根目錄下的測試音訊
        
    Returns:
        音訊文件的 Path 對象
    """
    # 從 tests/ 目錄向上找到專案根目錄
    project_root = Path(__file__).parent.parent
    return project_root / filename
