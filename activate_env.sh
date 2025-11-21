#!/bin/bash
# 快速啟動 Omnilingual ASR 環境

VENV_PATH="$HOME/omnilingual_env"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Omnilingual ASR 環境已啟動"
    # echo "可用指令："
    # echo "  python test_streaming_llm.py    # 測試串流"
    echo "  deactivate                  # 退出環境"
else
    echo "✗ 虛擬環境不存在: $VENV_PATH"
    echo "請先執行 ./setup_environment.sh"
fi
