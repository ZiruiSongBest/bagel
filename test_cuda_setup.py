#!/usr/bin/env python3
"""
测试CUDA设备配置脚本
用于验证修复后的分布式训练设置
"""

import os
import torch
import torch.distributed as dist
from datetime import timedelta

def test_cuda_setup():
    """测试CUDA设备配置"""
    print("=== CUDA设备配置测试 ===")
    
    # 检查CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    
    # 检查环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
    
    print(f"LOCAL_RANK: {local_rank}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 设置设备
    device_count = torch.cuda.device_count()
    device = local_rank % device_count
    print(f"分配的设备: {device}")
    
    # 设置CUDA设备
    torch.cuda.set_device(device)
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 测试CUDA内存
    print(f"设备{device}内存总量: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"设备{device}已分配内存: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")
    
    # 测试分布式初始化
    if world_size > 1:
        print("初始化分布式进程组...")
        try:
            dist.init_process_group("nccl", timeout=timedelta(minutes=5))
            print("分布式初始化成功")
            
            # 测试barrier同步
            print("测试barrier同步...")
            dist.barrier(device_ids=[device])
            print("barrier同步成功")
            
            # 清理
            dist.destroy_process_group()
            print("分布式进程组已清理")
            
        except Exception as e:
            print(f"分布式测试失败: {e}")
            return False
    
    print("=== 测试完成 ===")
    return True

if __name__ == "__main__":
    success = test_cuda_setup()
    exit(0 if success else 1)
