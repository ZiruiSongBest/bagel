#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 统一多模态生成示例

这个脚本提供了一个简单的入口来运行统一多模态生成示例
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_generation_example import main, load_bagel_model, UnifiedGenerationExample

def quick_start():
    """快速开始，使用默认设置"""
    print("🚀 统一多模态生成 - 快速开始")
    print("=" * 50)
    
    # 使用默认参数
    model_path = "models/BAGEL-7B-MoT"
    mode = 1  # 标准模式
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请确保已下载BAGEL模型到正确位置")
        print("下载地址: https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT")
        return False
    
    try:
        print("📥 正在加载模型...")
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
            model_path=model_path,
            mode=mode
        )
        
        print("🔧 创建推理器...")
        from inferencer import InterleaveInferencer
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        
        print("🎯 开始运行示例...")
        example = UnifiedGenerationExample(inferencer)
        
        # 运行一个简单的示例
        print("\n" + "="*50)
        print("🎨 运行简单的文本到图像生成示例...")
        example.example_1_simple_text_to_image()
        
        print("\n" + "="*50)
        print("🔍 分析原始token序列...")
        example.example_3_raw_token_analysis()
        
        print("\n✅ 快速示例完成！")
        print("💡 使用 'python unified_generation_example.py --help' 查看更多选项")
        
        return True
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_demo():
    """交互式演示"""
    print("🎮 统一多模态生成 - 交互式演示")
    print("=" * 50)
    
    # 简单的交互菜单
    while True:
        print("\n选择操作:")
        print("1. 快速开始 (默认设置)")
        print("2. 自定义设置运行")
        print("3. 仅测试模型加载")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            quick_start()
            break
        elif choice == "2":
            # 获取用户自定义参数
            model_path = input("模型路径 (默认: models/BAGEL-7B-MoT): ").strip()
            if not model_path:
                model_path = "models/BAGEL-7B-MoT"
            
            mode_input = input("加载模式 (1=标准, 2=NF4, 3=INT8, 默认: 1): ").strip()
            mode = int(mode_input) if mode_input in ["1", "2", "3"] else 1
            
            example_choice = input("示例选择 (all/1/2/3/4/5, 默认: all): ").strip()
            if not example_choice:
                example_choice = "all"
            
            # 构造命令行参数并运行
            sys.argv = [
                "unified_generation_example.py",
                "--model_path", model_path,
                "--mode", str(mode),
                "--example", example_choice
            ]
            main()
            break
        elif choice == "3":
            # 仅测试模型加载
            model_path = input("模型路径 (默认: models/BAGEL-7B-MoT): ").strip()
            if not model_path:
                model_path = "models/BAGEL-7B-MoT"
            
            try:
                print("🧪 测试模型加载...")
                model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
                    model_path=model_path,
                    mode=1
                )
                print("✅ 模型加载成功！")
                print(f"📊 特殊token数量: {len(new_token_ids)}")
                print(f"🔤 词汇表大小: {len(tokenizer)}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
            break
        elif choice == "4":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 如果有命令行参数，直接运行main函数
        main()
    else:
        # 否则运行交互式演示
        interactive_demo()
