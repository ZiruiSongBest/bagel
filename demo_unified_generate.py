#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像编辑Demo - 参考app.py的edit_image实现方式
"""

import os
import torch
import random
import numpy as np
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer


def set_seed(seed):
    """Set random seeds for reproducibility"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def load_bagel_model(model_path="models/BAGEL-7B-MoT", mode=1):
    """
    加载BAGEL模型，参考app.py的实现
    """
    print(f"正在加载BAGEL模型，路径: {model_path}")
    
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

    # 初始化空模型
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

    # 模型设备映射
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

    print("模型加载完成!")
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def edit_image(inferencer, image_path: str, prompt: str, show_thinking=False, 
               cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=0.0, 
               timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0, 
               cfg_renorm_type="text_channel", max_think_token_n=1024, 
               do_sample=False, text_temperature=0.3, seed=0):
    """
    图像编辑函数，参考app.py的edit_image实现
    
    Args:
        inferencer: InterleaveInferencer实例
        image_path: 输入图像路径
        prompt: 编辑提示
        其他参数与app.py中edit_image函数相同
    
    Returns:
        tuple: (编辑后的图像, 思考过程文本)
    """
    # 设置随机种子
    set_seed(seed)
    
    # 加载图像
    try:
        image = Image.open(image_path)
        print(f"成功加载图像: {image_path}, 尺寸: {image.size}")
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None, ""

    # 转换为numpy数组再转回PIL（如果需要）
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 转换为RGB格式
    image = pil_img2rgb(image)
    
    # 设置推理超参数
    inference_hyper = dict(
        max_think_token_n=max_think_token_n if show_thinking else 1024,
        do_sample=do_sample if show_thinking else False,
        text_temperature=text_temperature if show_thinking else 0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],  # End fixed at 1.0
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    print(f"开始图像编辑...")
    print(f"提示词: {prompt}")
    print(f"是否显示思考过程: {show_thinking}")
    print(f"推理参数: {inference_hyper}")
    
    # 调用inferencer进行图像编辑
    result = inferencer(image=image, text=prompt, think=show_thinking, **inference_hyper)
    
    return result["image"], result.get("text", "")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="图像编辑Demo")
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT",
                       help="模型路径")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                       help="加载模式: 1=标准, 2=NF4量化, 3=INT8量化")
    parser.add_argument("--image_path", type=str, 
                       default="/hdd_16T/Zirui/workspace/ATest/Bagel/assets/munchkin-cat-breed.jpg",
                       help="输入图像路径")
    parser.add_argument("--prompt", type=str, 
                       default="将这只猫咪变成一只穿着小礼服的优雅猫咪",
                       help="编辑提示词")
    parser.add_argument("--output_path", type=str, default="edited_cat.png",
                       help="输出图像路径")
    parser.add_argument("--show_thinking", action="store_true",
                       help="显示思考过程")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # CFG参数
    parser.add_argument("--cfg_text_scale", type=float, default=4.0,
                       help="文本CFG强度")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0,
                       help="图像CFG强度")
    parser.add_argument("--cfg_interval", type=float, default=0.0,
                       help="CFG应用间隔起始值")
    parser.add_argument("--timestep_shift", type=float, default=3.0,
                       help="时间步偏移")
    parser.add_argument("--num_timesteps", type=int, default=50,
                       help="时间步数")
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel",
                       choices=["global", "local", "text_channel"],
                       help="CFG重归一化类型")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("图像编辑Demo - 基于BAGEL模型")
    print("=" * 60)
    
    try:
        # 加载模型
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
            model_path=args.model_path,
            mode=args.mode
        )
        
        # 创建inferencer
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        
        print("\n开始图像编辑...")
        
        # 执行图像编辑
        edited_image, thinking_text = edit_image(
            inferencer=inferencer,
            image_path=args.image_path,
            prompt=args.prompt,
            show_thinking=args.show_thinking,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_interval=args.cfg_interval,
            timestep_shift=args.timestep_shift,
            num_timesteps=args.num_timesteps,
            cfg_renorm_type=args.cfg_renorm_type,
            seed=args.seed
        )
        
        # 保存结果
        if edited_image is not None:
            edited_image.save(args.output_path)
            print(f"\n✅ 编辑成功！")
            print(f"原始图像: {args.image_path}")
            print(f"编辑提示: {args.prompt}")
            print(f"输出图像: {args.output_path}")
            print(f"图像尺寸: {edited_image.size}")
            
            if args.show_thinking and thinking_text:
                print(f"\n🤔 思考过程:")
                print(thinking_text)
        else:
            print("❌ 图像编辑失败")
        
        print("\n" + "=" * 60)
        print("Demo运行完成!")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("1. 模型路径是否正确")
        print("2. 输入图像是否存在")
        print("3. 显存是否足够")
        print("4. 所有依赖是否已安装")


if __name__ == "__main__":
    main()