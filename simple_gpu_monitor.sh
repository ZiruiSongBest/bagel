#!/bin/bash

# 简化版GPU监控脚本
# 每30秒检查一次GPU状态，当4张显卡都空闲时启动训练

echo "开始监控GPU状态..."

while true; do
    # 检查4张GPU的内存使用情况
    gpu0_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    gpu1_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    gpu2_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    gpu3_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    
    # 显示当前GPU状态
    echo "$(date '+%H:%M:%S') - GPU状态: GPU0:${gpu0_mem}MB GPU1:${gpu1_mem}MB GPU2:${gpu2_mem}MB GPU3:${gpu3_mem}MB"
    
    # 检查是否所有GPU都空闲（内存使用小于500MB）
    if [ "$gpu0_mem" -lt 500 ] && [ "$gpu1_mem" -lt 500 ] && [ "$gpu2_mem" -lt 500 ] && [ "$gpu3_mem" -lt 500 ]; then
        echo "所有GPU都空闲，启动训练..."
        bash ./unitrain_modelparallel.sh
        echo "训练完成，退出监控"
        break
    else
        echo "有GPU正在使用中，30秒后再次检查..."
        sleep 30
    fi
done
