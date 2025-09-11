#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自回归交错生成模型推理示例

这个文件提供了多种使用训练好的统一模型进行推理的示例。
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image
import logging

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from inference_unified_autoregressive import create_inference_engine

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_text_to_image():
    """示例1: 文本转图像生成"""
    print("=== 示例1: 文本转图像生成 ===")
    
    # 检查点路径（请根据您的实际路径修改）
    checkpoint_path = "/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点路径不存在: {checkpoint_path}")
        print("请修改为您的实际检查点路径")
        return
    
    try:
        # 创建推理引擎
        engine = create_inference_engine(checkpoint_path)
        
        # 文本转图像的提示
        prompts = [
            "画一只可爱的橘猫在花园里玩耍，阳光明媚，水彩画风格。",
            "创作一幅山水画，描绘清晨的湖泊，远山如黛，薄雾缭绕。",
            "设计一个现代简约的客厅，有大落地窗和舒适的沙发。"
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"\n处理提示 {i+1}: {prompt}")
            
            results = engine.autoregressive_generate(
                prompt=prompt,
                max_length=300,
                do_sample=True,
                temperature=0.8,
                image_shapes=(1024, 1024),
                save_intermediate=True,
                output_dir=f"example1_output_{i+1}"
            )
            
            print(f"生成完成，共 {len(results)} 项结果")
            for j, item in enumerate(results):
                if isinstance(item, str):
                    print(f"  [{j}] 文本: {item[:100]}")
                elif isinstance(item, Image.Image):
                    print(f"  [{j}] 图像: {item.size}")
    
    except Exception as e:
        print(f"示例1执行失败: {e}")


def example_image_editing():
    """示例2: 图像编辑链"""
    print("\n=== 示例2: 图像编辑链 ===")
    
    checkpoint_path = "/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点路径不存在: {checkpoint_path}")
        return
    
    # 检查是否有测试图像
    test_image_paths = [
        "/workspace/bagel/test_images/meme.jpg",
        "/workspace/bagel/test_images/octupusy.jpg", 
        "/workspace/bagel/test_images/women.jpg",
        "/workspace/bagel/assets/munchkin-cat-breed.jpg"
    ]
    
    input_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            input_image_path = path
            break
    
    if input_image_path is None:
        print("未找到测试图像，跳过图像编辑示例")
        return
    
    try:
        # 创建推理引擎
        engine = create_inference_engine(checkpoint_path)
        
        # 加载输入图像
        input_image = Image.open(input_image_path).convert('RGB')
        print(f"加载输入图像: {input_image.size}")
        
        # 编辑指令序列
        edit_instructions = [
            "添加一个彩虹背景",
            "让主体变得更加生动有趣", 
            "应用卡通风格的效果"
        ]
        
        results = engine.image_editing_chain(
            input_image=input_image,
            edit_instructions=edit_instructions,
            image_shapes=(1024, 1024),
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            num_timesteps=50
        )
        
        print(f"编辑完成，共 {len(results)} 项结果")
        
        # 保存结果
        output_dir = "example2_image_editing"
        os.makedirs(output_dir, exist_ok=True)
        
        image_count = 0
        for i, item in enumerate(results):
            if isinstance(item, Image.Image):
                save_path = os.path.join(output_dir, f"edited_step_{image_count}.png")
                item.save(save_path)
                print(f"保存编辑结果: {save_path}")
                image_count += 1
            elif isinstance(item, str):
                print(f"  步骤: {item}")
    
    except Exception as e:
        print(f"示例2执行失败: {e}")


def example_multi_step_story():
    """示例3: 多步骤故事生成"""
    print("\n=== 示例3: 多步骤故事生成 ===")
    
    checkpoint_path = "/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点路径不存在: {checkpoint_path}")
        return
    
    try:
        # 创建推理引擎
        engine = create_inference_engine(checkpoint_path)
        
        # 分步骤的故事指令
        story_steps = [
            "从前有一个小镇，镇上有一座古老的图书馆。请画出这个图书馆的外观。",
            "图书馆里住着一位和蔼的老图书管理员。请描述一下这位管理员的样子。",
            "一天，一个小女孩来到图书馆寻找魔法书。请画出小女孩在书架间寻找的场景。",
            "小女孩找到了一本发光的魔法书。请展示这本神奇的书。",
            "当她打开书时，书中的插图活了过来。请创作这个magical moment的画面。"
        ]
        
        results = engine.step_by_step_generate(
            instructions=story_steps,
            image_shapes=(1024, 1024),
            max_length=200,
            do_sample=True,
            temperature=0.7,
            cfg_text_scale=3.5
        )
        
        print(f"故事生成完成，共 {len(results)} 项结果")
        
        # 保存故事
        output_dir = "example3_story"
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建故事文档
        story_content = []
        image_count = 0
        
        for i, item in enumerate(results):
            if isinstance(item, str):
                story_content.append(f"第{i//2 + 1}章: {item}\n")
            elif isinstance(item, Image.Image):
                image_filename = f"story_image_{image_count}.png"
                image_path = os.path.join(output_dir, image_filename)
                item.save(image_path)
                story_content.append(f"[插图: {image_filename}]\n")
                print(f"保存故事插图: {image_path}")
                image_count += 1
        
        # 保存故事文本
        story_file = os.path.join(output_dir, "story.txt")
        with open(story_file, 'w', encoding='utf-8') as f:
            f.write("=== 自动生成的图文故事 ===\n\n")
            f.writelines(story_content)
        
        print(f"故事已保存到: {story_file}")
    
    except Exception as e:
        print(f"示例3执行失败: {e}")


def example_interactive_mode():
    """示例4: 交互式生成"""
    print("\n=== 示例4: 交互式生成模式 ===")
    
    checkpoint_path = "/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点路径不存在: {checkpoint_path}")
        return
    
    try:
        # 创建推理引擎
        engine = create_inference_engine(checkpoint_path)
        
        print("进入交互式生成模式。输入 'quit' 退出。")
        print("您可以输入文本描述，系统将自动决定生成文本还是图像。")
        print("示例：'画一只猫' 或 '描述一下春天的景色'\n")
        
        session_count = 0
        
        while True:
            try:
                user_input = input("请输入您的提示: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用，再见！")
                    break
                
                if not user_input:
                    continue
                
                session_count += 1
                print(f"\n[会话 {session_count}] 正在处理: {user_input}")
                
                results = engine.autoregressive_generate(
                    prompt=user_input,
                    max_length=300,
                    do_sample=True,
                    temperature=0.8,
                    image_shapes=(512, 512),  # 使用较小尺寸以加快速度
                    save_intermediate=True,
                    output_dir=f"interactive_session_{session_count}"
                )
                
                print(f"生成完成！结果:")
                for i, item in enumerate(results):
                    if isinstance(item, str):
                        print(f"  文本: {item}")
                    elif isinstance(item, Image.Image):
                        print(f"  图像: {item.size} (已保存到 interactive_session_{session_count}/)")
                
                print()
                
            except KeyboardInterrupt:
                print("\n检测到中断，退出交互模式")
                break
            except Exception as e:
                print(f"处理请求时出错: {e}")
                continue
    
    except Exception as e:
        print(f"示例4执行失败: {e}")


def main():
    """运行所有示例"""
    print("🚀 自回归交错生成模型推理示例")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ 检测到CUDA设备: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  未检测到CUDA设备，将使用CPU（可能很慢）")
    
    print("\n可用示例:")
    print("1. 文本转图像生成")
    print("2. 图像编辑链")  
    print("3. 多步骤故事生成")
    print("4. 交互式生成模式")
    print("5. 运行所有示例")
    
    choice = input("\n请选择要运行的示例 (1-5): ").strip()
    
    if choice == "1":
        example_text_to_image()
    elif choice == "2":
        example_image_editing()
    elif choice == "3":
        example_multi_step_story()
    elif choice == "4":
        example_interactive_mode()
    elif choice == "5":
        example_text_to_image()
        example_image_editing()
        example_multi_step_story()
    else:
        print("无效选择，退出")
        return
    
    print("\n示例运行完成！")


if __name__ == "__main__":
    main()
