#!/bin/bash
# Omnilingual ASR 環境設定腳本（WSL2 + GPU 支援）

set -e  # 遇到錯誤立即停止

echo "=================================="
echo "Omnilingual ASR 環境設定"
echo "=================================="
echo ""

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 檢查函數
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 已安裝"
        return 0
    else
        echo -e "${RED}✗${NC} $1 未安裝"
        return 1
    fi
}

# Step 1: 檢查系統需求
echo "Step 1: 檢查系統需求"
echo "-------------------"

# 檢查 Python 版本
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "${GREEN}✓${NC} Python 版本: $(python3 --version)"

    # 檢查是否 >= 3.10
    REQUIRED_VERSION="3.10"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        echo -e "${GREEN}✓${NC} Python 版本符合需求 (>= 3.10)"
    else
        echo -e "${YELLOW}⚠${NC}  Python 版本建議升級到 3.10+ (當前: $PYTHON_VERSION)"
    fi
else
    echo -e "${RED}✗${NC} Python3 未安裝！"
    echo "請先安裝 Python 3.10+："
    echo "  sudo apt update"
    echo "  sudo apt install python3.10 python3.10-venv python3-pip"
    exit 1
fi

# 檢查 GPU
echo ""
echo "Step 2: 檢查 GPU 支援"
echo "-------------------"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU 已檢測到"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "  GPU: $line"
    done
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC}  未檢測到 NVIDIA GPU 或驅動未安裝"
    echo "  將使用 CPU 模式"
    echo "  如需 GPU 支援，請參考 wsl_gpu_setup_guide.md"
    GPU_AVAILABLE=false
fi

# Step 3: 安裝系統依賴
echo ""
echo "Step 3: 安裝系統依賴"
echo "-------------------"

echo "更新套件清單..."
sudo apt update -qq

SYSTEM_DEPS=(
    "libsndfile1"
    "ffmpeg"
    "build-essential"
    "python3-dev"
    "git"
)

for dep in "${SYSTEM_DEPS[@]}"; do
    if dpkg -s $dep &> /dev/null; then
        echo -e "${GREEN}✓${NC} $dep 已安裝"
    else
        echo "安裝 $dep..."
        sudo apt install -y $dep
    fi
done

# Step 4: 建立虛擬環境
echo ""
echo "Step 4: 建立 Python 虛擬環境"
echo "----------------------------"

VENV_PATH="$HOME/omnilingual_env"

if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}⚠${NC}  虛擬環境已存在: $VENV_PATH"
    read -p "是否刪除並重建？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        echo "已刪除舊環境"
    else
        echo "使用現有環境"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "建立虛擬環境於: $VENV_PATH"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓${NC} 虛擬環境建立完成"
fi

# 啟動虛擬環境
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓${NC} 虛擬環境已啟動"

# Step 5: 升級 pip
echo ""
echo "Step 5: 升級 pip"
echo "---------------"
pip install --upgrade pip setuptools wheel -q
echo -e "${GREEN}✓${NC} pip 已升級到最新版本"

# Step 6: 安裝 PyTorch
echo ""
echo "Step 6: 安裝 PyTorch"
echo "-------------------"

if $GPU_AVAILABLE; then
    echo "安裝 PyTorch (CUDA 12.1 版本)..."
    echo "這可能需要幾分鐘..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # 驗證 CUDA
    python3 << EOF
import torch
if torch.cuda.is_available():
    print("\n${GREEN}✓${NC} PyTorch CUDA 支援已啟用")
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n${YELLOW}⚠${NC}  PyTorch 已安裝但 CUDA 不可用")
    print("  請檢查 CUDA Toolkit 是否已安裝")
EOF
else
    echo "安裝 PyTorch (CPU 版本)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo -e "${GREEN}✓${NC} PyTorch (CPU) 安裝完成"
fi

# Step 7: 安裝 Omnilingual ASR
echo ""
echo "Step 7: 安裝 Omnilingual ASR"
echo "---------------------------"

# 檢查是否在專案目錄
if [ -f "pyproject.toml" ]; then
    echo "偵測到專案根目錄，安裝開發模式..."
    pip install -e ".[dev]"
    echo -e "${GREEN}✓${NC} Omnilingual ASR 安裝完成 (開發模式)"
else
    echo -e "${RED}✗${NC} 找不到 pyproject.toml"
    echo "請在專案根目錄執行此腳本"
    exit 1
fi

# Step 8: 驗證安裝
echo ""
echo "Step 8: 驗證安裝"
echo "---------------"

python3 << 'VERIFY_EOF'
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: 未安裝")
        return False

print("\n核心套件檢查：")
print("-" * 40)

all_ok = True
all_ok &= check_import('torch', 'PyTorch')
all_ok &= check_import('fairseq2', 'fairseq2')
all_ok &= check_import('omnilingual_asr', 'Omnilingual ASR')

print("\n額外套件檢查：")
print("-" * 40)
check_import('numpy', 'NumPy')
check_import('soundfile', 'soundfile')

# GPU 檢查
print("\nGPU 支援檢查：")
print("-" * 40)
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA 可用")
    print(f"  版本: {torch.version.cuda}")
    print(f"  裝置數: {torch.cuda.device_count()}")
    print(f"  GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"  記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ CUDA 不可用（將使用 CPU）")

if not all_ok:
    print("\n⚠ 部分套件安裝失敗，請檢查錯誤訊息")
    sys.exit(1)
VERIFY_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓${NC} 所有套件驗證通過！"
else
    echo ""
    echo -e "${RED}✗${NC} 驗證失敗，請檢查上方錯誤訊息"
    exit 1
fi

# Step 9: 建立啟動腳本
echo ""
echo "Step 9: 建立快速啟動腳本"
echo "------------------------"

cat > activate_env.sh << 'ACTIVATE_EOF'
#!/bin/bash
# 快速啟動 Omnilingual ASR 環境

VENV_PATH="$HOME/omnilingual_env"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Omnilingual ASR 環境已啟動"
    echo ""
    echo "可用指令："
    echo "  python test_gpu.py          # 測試 GPU"
    echo "  python test_streaming.py    # 測試串流"
    echo "  deactivate                  # 退出環境"
else
    echo "✗ 虛擬環境不存在: $VENV_PATH"
    echo "請先執行 ./setup_environment.sh"
fi
ACTIVATE_EOF

chmod +x activate_env.sh
echo -e "${GREEN}✓${NC} 啟動腳本已建立: ./activate_env.sh"

# Step 10: 建立測試腳本
echo ""
echo "Step 10: 建立測試腳本"
echo "--------------------"

cat > test_installation.py << 'TEST_EOF'
#!/usr/bin/env python3
"""測試 Omnilingual ASR 安裝"""

import torch
import numpy as np
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

def test_basic_inference():
    """基本推理測試"""
    print("="*60)
    print("Omnilingual ASR 安裝測試")
    print("="*60)

    # 檢查裝置
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"\n⚠ 使用 CPU（GPU 不可用）")

    # 載入模型（使用最小模型）
    print("\n正在載入模型...")
    try:
        pipeline = ASRInferencePipeline(
            model_card="omniASR_CTC_300M",  # 最小模型
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        print("✓ 模型載入成功")
    except Exception as e:
        print(f"✗ 模型載入失敗: {e}")
        return False

    # 產生測試音訊（3 秒白噪音）
    print("\n正在測試推理...")
    sample_rate = 16000
    audio = np.random.randn(3 * sample_rate).astype(np.float32) * 0.1

    try:
        import time
        start = time.time()

        result = pipeline.transcribe(
            inp=[{"waveform": torch.from_numpy(audio), "sample_rate": sample_rate}],
            batch_size=1,
        )

        inference_time = time.time() - start

        print(f"✓ 推理成功")
        print(f"  處理時間: {inference_time:.3f}s")
        print(f"  RTF: {inference_time / 3.0:.3f}")
        print(f"  結果: {result[0][:50]}...")

        if device == "cuda":
            print(f"\nGPU 記憶體使用:")
            print(f"  已分配: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        print("\n" + "="*60)
        print("✓ 測試通過！環境設定成功！")
        print("="*60)
        return True

    except Exception as e:
        print(f"✗ 推理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_inference()
    exit(0 if success else 1)
TEST_EOF

chmod +x test_installation.py
echo -e "${GREEN}✓${NC} 測試腳本已建立: ./test_installation.py"

# 完成
echo ""
echo "=================================="
echo "✓ 環境設定完成！"
echo "=================================="
echo ""
echo "下一步："
echo "  1. 啟動環境："
echo "     source ~/omnilingual_env/bin/activate"
echo "     # 或"
echo "     ./activate_env.sh"
echo ""
echo "  2. 測試安裝："
echo "     python test_installation.py"
echo ""
echo "  3. 開始開發："
echo "     python test_gpu.py              # 測試 GPU"
echo "     python test_streaming.py         # 測試串流"
echo ""
echo "環境資訊："
echo "  虛擬環境: $VENV_PATH"
echo "  Python: $(python3 --version)"
if $GPU_AVAILABLE; then
    echo "  GPU: 已啟用"
else
    echo "  GPU: 未啟用 (CPU 模式)"
fi
echo ""
