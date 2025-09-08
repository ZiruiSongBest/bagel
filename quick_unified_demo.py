#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示 inferencer.unified_generate() 方法的用法

这是一个简化的示例，展示如何使用 unified_generate 实现文本和图像的混合生成。
"""

import torch
from PIL import Image
from inferencer import InterleaveInferencer

# 假设你已经有了加载好的模型组件
# 这里提供使用 unified_generate 的核心代码示例

def quick_demo_unified_generate(inferencer):
    """快速演示 unified_generate 的基本用法"""
    
    print("=== unified_generate 基本用法演示 ===")
    
    # 示例1: 简单的文本转图像
    print("\n1. 简单文本转图像:")
    prompt1 = "画一只可爱的小猫在草地上玩耍"
    
    result1 = inferencer.unified_generate(
        input_text=prompt1,
        max_length=200,
        do_sample=True,
        temperature=0.8,
        image_shapes=(512, 512),
        return_raw_tokens=False
    )
    
    print(f"结果类型: {[type(item).__name__ for item in result1]}")
    for i, item in enumerate(result1):
        if isinstance(item, str):
            print(f"文本: {item}")
        elif isinstance(item, Image.Image):
            print(f"图像: {item.size}")
            item.save(f"quick_demo_cat_{i}.png")
    
    # 示例2: 更复杂的多模态生成
    print("\n2. 复杂多模态生成:")
    prompt2 = """描述一个美丽的花园场景：

花园里有各种鲜花盛开，请画出这个花园的全景图。

在花园的中央有一个小池塘，池塘里有金鱼游来游去，请展示池塘的特写画面。

花园的角落里有一个小亭子，人们可以在那里休息，请画出这个亭子的图像。"""
    
    result2 = inferencer.unified_generate(
        input_text=prompt2,
        max_length=500,
        do_sample=True,
        temperature=0.7,
        image_shapes=(768, 768),
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        return_raw_tokens=False
    )
    
    print(f"复杂生成结果包含 {len(result2)} 个元素:")
    for i, item in enumerate(result2):
        if isinstance(item, str):
            print(f"文本 {i}: {item[:100]}...")
        elif isinstance(item, Image.Image):
            filename = f"garden_scene_{i}.png"
            item.save(filename)
            print(f"图像 {i}: 已保存为 {filename}")
    
    # 示例3: 获取原始token序列
    print("\n3. 获取原始token序列:")
    prompt3 = "画一朵红玫瑰"
    
    raw_tokens = inferencer.unified_generate(
        input_text=prompt3,
        max_length=100,
        do_sample=False,
        return_raw_tokens=True  # 返回原始token而不是解析后的结果
    )
    
    print(f"原始token序列长度: {len(raw_tokens)}")
    print(f"前10个token: {raw_tokens[:10]}")
    
    # 分析特殊token
    special_token_names = {
        inferencer.new_token_ids.get('start_of_image', -1): 'IMAGE_START',
        inferencer.new_token_ids.get('end_of_image', -1): 'IMAGE_END',
        inferencer.new_token_ids.get('eos_token_id', -1): 'EOS',
        inferencer.new_token_ids.get('bos_token_id', -1): 'BOS'
    }
    
    print("特殊token检测:")
    for i, token_id in enumerate(raw_tokens):
        if token_id in special_token_names:
            print(f"位置 {i}: {special_token_names[token_id]}")
    
    return result1, result2, raw_tokens


def demo_unified_generate_parameters():
    """演示 unified_generate 的各种参数设置"""
    
    print("\n=== unified_generate 参数详解 ===")
    
    # 参数说明
    parameters_info = {
        "input_text": "输入的文本提示",
        "max_length": "最大生成长度（token数量）",
        "do_sample": "是否使用随机采样（True）还是贪心搜索（False）",
        "temperature": "采样温度，控制随机性（越大越随机）",
        "image_shapes": "生成图像的尺寸，如 (1024, 1024)",
        "cfg_text_scale": "文本引导强度（CFG scale）",
        "cfg_img_scale": "图像引导强度",
        "cfg_interval": "CFG应用的时间步区间",
        "timestep_shift": "时间步偏移参数",
        "num_timesteps": "去噪步骤数量",
        "cfg_renorm_min": "CFG重归一化最小值",
        "cfg_renorm_type": "CFG重归一化类型",
        "enable_taylorseer": "是否启用TaylorSeer加速",
        "return_raw_tokens": "是否返回原始token序列而不是解析后的结果"
    }
    
    print("主要参数说明:")
    for param, description in parameters_info.items():
        print(f"  {param}: {description}")
    
    # 推荐的参数组合
    print("\n推荐的参数组合:")
    
    configs = {
        "高质量图像生成": {
            "image_shapes": (1024, 1024),
            "cfg_text_scale": 6.0,
            "cfg_img_scale": 2.0,
            "num_timesteps": 100,
            "do_sample": False
        },
        "快速生成": {
            "image_shapes": (512, 512),
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
            "num_timesteps": 20,
            "do_sample": True,
            "temperature": 0.8
        },
        "创意生成": {
            "image_shapes": (768, 768),
            "cfg_text_scale": 3.0,
            "cfg_img_scale": 1.2,
            "do_sample": True,
            "temperature": 1.0
        }
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name}:")
        for param, value in config.items():
            print(f"  {param}: {value}")


def usage_example_code():
    """显示使用 unified_generate 的示例代码"""
    
    example_code = '''
# 基本用法示例
result = inferencer.unified_generate(
    input_text="画一只蓝色的鸟在树枝上",
    max_length=300,
    do_sample=True,
    temperature=0.8,
    image_shapes=(512, 512),
    return_raw_tokens=False
)

# 处理结果
for i, item in enumerate(result):
    if isinstance(item, str):
        print(f"生成的文本: {item}")
    elif isinstance(item, Image.Image):
        print(f"生成的图像尺寸: {item.size}")
        item.save(f"generated_image_{i}.png")

# 高质量图像生成
high_quality_result = inferencer.unified_generate(
    input_text="一幅详细的风景画，包含山脉、湖泊和森林",
    max_length=400,
    image_shapes=(1024, 1024),
    cfg_text_scale=6.0,
    cfg_img_scale=2.0,
    num_timesteps=100,
    do_sample=False,
    return_raw_tokens=False
)

# 获取原始token进行分析
raw_tokens = inferencer.unified_generate(
    input_text="画一个红苹果",
    max_length=200,
    return_raw_tokens=True
)
print(f"生成了 {len(raw_tokens)} 个token")
'''
    
    print("\n=== 使用示例代码 ===")
    print(example_code)


if __name__ == "__main__":
    print("unified_generate() 方法使用指南")
    print("=" * 50)
    
    # 显示参数说明
    demo_unified_generate_parameters()
    
    # 显示示例代码
    usage_example_code()
    
    print("\n" + "=" * 50)
    print("说明:")
    print("1. 这个脚本展示了 unified_generate 的用法")
    print("2. 要运行实际的生成，需要先加载模型")
    print("3. 请参考 unified_generation_example.py 了解完整的模型加载流程")
    print("4. 或者运行 demo_unified_generate.py 进行完整演示")
