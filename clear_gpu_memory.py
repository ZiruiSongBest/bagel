#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU内存清理脚本
用于清理GPU缓存，释放显存
"""

import torch
import gc
import psutil
import os
import subprocess

def clear_gpu_memory():
    """清理GPU内存"""
    print("开始清理GPU内存...")
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        print(f"检测到 {torch.cuda.device_count()} 张GPU")
        
        # 清理每张GPU的缓存
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        print("PyTorch GPU缓存已清理")
    
    # Python垃圾回收
    gc.collect()
    print("Python垃圾回收完成")
    
    # 显示当前GPU使用情况
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, check=True)
        print("\n当前GPU内存使用情况:")
        print("GPU_ID, 已使用(MB), 总计(MB)")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                gpu_id, used, total = line.split(', ')
                used_gb = float(used) / 1024
                total_gb = float(total) / 1024
                usage_percent = (float(used) / float(total)) * 100
                print(f"GPU {gpu_id}: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_percent:.1f}%)")
    except subprocess.CalledProcessError:
        print("无法获取GPU信息")

if __name__ == "__main__":
    clear_gpu_memory()
