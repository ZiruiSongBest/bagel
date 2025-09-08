#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一多模态生成的使用示例和测试接口

这个文件展示了如何使用修改后的InterleaveInferencer来实现真正的统一多模态生成，
其中单个model.generate()调用可以产生包含文本和图像token的序列。
"""

from typing import List, Union, Dict, Any
import torch
from PIL import Image
import json

import os
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer


class UnifiedGenerationExample:
    """统一生成示例类，展示如何使用新的接口"""
    
    def __init__(self, inferencer):
        """
        Args:
            inferencer: 修改后的InterleaveInferencer实例
        """
        self.inferencer = inferencer
    
    def example_1_simple_text_to_image(self):
        """示例1: 简单的文本到图像生成"""
        print("=== 示例1: 简单文本到图像生成 ===")
        
        # 创建包含图像占位符的提示
        prompt = "请画一只可爱的小猫咪。"
        
        # 使用统一生成接口
        result = self.inferencer.unified_generate(
            input_text=prompt,
            max_length=200,
            do_sample=True,
            temperature=0.8,
            image_shapes=(512, 512),
            return_raw_tokens=False  # 返回解析后的结果
        )
        
        print(f"生成结果类型: {[type(item).__name__ for item in result]}")
        for i, item in enumerate(result):
            if isinstance(item, str):
                print(f"文本片段 {i}: {item}")
            elif isinstance(item, Image.Image):
                print(f"图像 {i}: {item.size}")
                # 可以保存图像
                item.save(f"generated_image_example1_{i}.png")
        
        return result
    
    def example_2_interleaved_generation(self):
        """示例2: 交替的文本-图像生成"""
        print("\n=== 示例2: 交替文本-图像生成 ===")
        
        # 创建更复杂的提示，包含多个图像占位符
        prompt = "我想要一个故事：从前有一座山，山上有一只小兔子。让我看看这只兔子的样子。然后这只兔子遇到了一只狐狸，让我看看它们一起玩耍的场景。"
        
        result = self.inferencer.unified_generate(
            input_text=prompt,
            max_length=500,
            do_sample=True,
            temperature=0.7,
            image_shapes=(768, 768),
            return_raw_tokens=False
        )
        
        print(f"生成了 {len(result)} 个元素")
        for i, item in enumerate(result):
            if isinstance(item, str):
                print(f"文本 {i}: {item[:100]}...")
            elif isinstance(item, Image.Image):
                print(f"图像 {i}: {item.size}")
                item.save(f"generated_image_example2_{i}.png")
        
        return result
    
    def example_3_raw_token_analysis(self):
        """示例3: 分析原始token序列"""
        print("\n=== 示例3: 原始Token序列分析 ===")
        
        prompt = "画一幅美丽的风景画，包含山和水。"
        
        # 获取原始token序列
        raw_tokens = self.inferencer.unified_generate(
            input_text=prompt,
            max_length=300,
            do_sample=False,  # 使用确定性生成便于分析
            return_raw_tokens=True  # 返回原始token
        )
        
        print(f"生成了 {len(raw_tokens)} 个token")
        print("Token序列:")
        
        # 分析token序列
        special_tokens = {
            'img_start_token_id': '图像开始',
            'img_end_token_id': '图像结束', 
            'eos_token_id': '序列结束',
            'bos_token_id': '序列开始'
        }
        
        for i, token_id in enumerate(raw_tokens[:50]):  # 只显示前50个token
            if token_id in [self.inferencer.new_token_ids.get(k) for k in special_tokens.keys()]:
                # 找到对应的特殊token名称
                for special_name, token_value in self.inferencer.new_token_ids.items():
                    if token_value == token_id:
                        print(f"Token {i}: {token_id} -> [{special_tokens.get(special_name, special_name)}]")
                        break
            else:
                # 普通token，解码为文本
                try:
                    text = self.inferencer.tokenizer.decode([token_id])
                    print(f"Token {i}: {token_id} -> '{text}'")
                except:
                    print(f"Token {i}: {token_id} -> [无法解码]")
        
        if len(raw_tokens) > 50:
            print(f"... (省略了 {len(raw_tokens) - 50} 个token)")
        
        return raw_tokens
    
    def example_4_custom_cfg_parameters(self):
        """示例4: 自定义CFG参数的图像生成"""
        print("\n=== 示例4: 自定义CFG参数生成 ===")
        
        prompt = "一个超现实主义风格的梦境场景。"
        
        result = self.inferencer.unified_generate(
            input_text=prompt,
            max_length=250,
            image_shapes=(1024, 1024),
            cfg_text_scale=6.0,  # 更强的文本引导
            cfg_img_scale=2.0,   # 更强的图像引导
            timestep_shift=4.0,  # 调整时间步长
            num_timesteps=100,   # 更多的去噪步骤
            return_raw_tokens=False
        )
        
        for i, item in enumerate(result):
            if isinstance(item, Image.Image):
                print(f"高质量图像 {i}: {item.size}")
                item.save(f"generated_high_quality_{i}.png")
        
        return result
    
    def create_structured_prompt(self, story_elements: List[Dict[str, Any]]) -> str:
        """
        创建结构化的提示，支持复杂的文本-图像交替模式
        
        Args:
            story_elements: 故事元素列表，每个元素包含type和content
                          type可以是'text'或'image_request'
        """
        prompt_parts = []
        
        for element in story_elements:
            if element['type'] == 'text':
                prompt_parts.append(element['content'])
            elif element['type'] == 'image_request':
                # 添加图像请求的特殊标记
                img_prompt = f"[请生成图像: {element['content']}]"
                prompt_parts.append(img_prompt)
        
        return " ".join(prompt_parts)
    
    def example_5_structured_storytelling(self):
        """示例5: 结构化故事叙述"""
        print("\n=== 示例5: 结构化故事生成 ===")
        
        # 定义故事结构
        story_elements = [
            {"type": "text", "content": "从前，在一个魔法森林里"},
            {"type": "image_request", "content": "魔法森林的全景图"},
            {"type": "text", "content": "住着一只会说话的小龙"},
            {"type": "image_request", "content": "可爱的小龙在森林中"},
            {"type": "text", "content": "它每天都在寻找传说中的彩虹花"},
            {"type": "image_request", "content": "神秘的彩虹花在阳光下闪闪发光"}
        ]
        
        # 创建结构化提示
        structured_prompt = self.create_structured_prompt(story_elements)
        print(f"结构化提示: {structured_prompt}")
        
        result = self.inferencer.unified_generate(
            input_text=structured_prompt,
            max_length=600,
            do_sample=True,
            temperature=0.8,
            image_shapes=(768, 512),
            return_raw_tokens=False
        )
        
        # 保存结果
        story_output = {"elements": []}
        for i, item in enumerate(result):
            if isinstance(item, str):
                story_output["elements"].append({"type": "text", "content": item})
                print(f"故事文本 {i}: {item}")
            elif isinstance(item, Image.Image):
                filename = f"story_image_{i}.png"
                item.save(filename)
                story_output["elements"].append({"type": "image", "filename": filename})
                print(f"故事图像 {i}: 保存为 {filename}")
        
        # 保存故事元数据
        with open("generated_story.json", "w", encoding="utf-8") as f:
            json.dump(story_output, f, ensure_ascii=False, indent=2)
        
        return result
    
    def run_all_examples(self):
        """运行所有示例"""
        print("开始运行统一多模态生成示例...")
        print("=" * 60)
        
        try:
            self.example_1_simple_text_to_image()
            self.example_2_interleaved_generation()
            self.example_3_raw_token_analysis()
            self.example_4_custom_cfg_parameters()
            self.example_5_structured_storytelling()
            
            print("\n" + "=" * 60)
            print("所有示例运行完成！")
            
        except Exception as e:
            print(f"运行示例时出错: {e}")
            import traceback
            traceback.print_exc()


def load_bagel_model(model_path="models/BAGEL-7B-MoT", mode=1):
    """
    根据app.py的加载逻辑，加载BAGEL模型
    
    Args:
        model_path: 模型路径
        mode: 加载模式 (1=normal, 2=NF4, 3=INT8)
    
    Returns:
        tuple: (model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    """

    
    print(f"开始加载BAGEL模型，路径: {model_path}")
    
    # 加载配置
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # 初始化模型结构
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # 加载tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # 创建transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # 设备映射和模型加载
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # 根据模式加载模型权重
    if mode == 1:
        print("使用标准模式加载模型...")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif mode == 2:  # NF4
        print("使用NF4量化模式加载模型...")
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=False, 
            bnb_4bit_quant_type="nf4"
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif mode == 3:  # INT8
        print("使用INT8量化模式加载模型...")
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True, 
            torch_dtype=torch.float32
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError(f"模式 {mode} 未实现")
    
    print("模型加载完成！")
    print(f"新增的特殊token ID: {new_token_ids}")
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def main():
    """主函数，展示如何初始化和使用统一生成接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统一多模态生成示例")
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT",
                       help="模型路径")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                       help="加载模式: 1=标准, 2=NF4量化, 3=INT8量化")
    parser.add_argument("--example", type=str, default="all",
                       choices=["all", "1", "2", "3", "4", "5"],
                       help="运行哪个示例 (all=所有示例)")
    
    args = parser.parse_args()
    
    print("统一多模态生成示例")
    print("=" * 60)
    
    try:
        # 加载模型和组件
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
            model_path=args.model_path,
            mode=args.mode
        )
        
        # 创建inferencer实例
        from inferencer import InterleaveInferencer
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        
        # 创建示例类
        example = UnifiedGenerationExample(inferencer)
        
        # 根据用户选择运行特定示例
        if args.example == "all":
            example.run_all_examples()
        elif args.example == "1":
            example.example_1_simple_text_to_image()
        elif args.example == "2":
            example.example_2_interleaved_generation()
        elif args.example == "3":
            example.example_3_raw_token_analysis()
        elif args.example == "4":
            example.example_4_custom_cfg_parameters()
        elif args.example == "5":
            example.example_5_structured_storytelling()
        
        print("\n" + "=" * 60)
        print("示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保:")
        print("1. 模型路径正确")
        print("2. 模型文件完整")
        print("3. 显存足够")
        print("4. 所有依赖已安装")


if __name__ == "__main__":
    main()
