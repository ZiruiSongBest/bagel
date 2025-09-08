#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像编辑功能演示Demo

这个文件展示了如何使用BAGEL模型的unified generation功能进行图像编辑。
通过输入一张图像和编辑提示词，生成编辑后的新图像。
"""

import os
import torch
from PIL import Image
from typing import List, Union, Optional

# 导入必要的模块
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights


class ImageEditingDemo:
    """图像编辑演示类"""
    
    def __init__(self, inferencer):
        """
        Args:
            inferencer: InterleaveInferencer实例
        """
        self.inferencer = inferencer
    
    def edit_image_with_prompt(
        self, 
        input_image_path: str,
        edit_prompt: str,
        output_path: Optional[str] = None,
        image_size: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        enable_thinking: bool = True
    ) -> Image.Image:
        """
        使用提示词编辑图像
        
        Args:
            input_image_path: 输入图像的路径
            edit_prompt: 编辑提示词
            output_path: 输出图像保存路径（可选）
            image_size: 生成图像的尺寸
            cfg_text_scale: 文本CFG缩放因子
            cfg_img_scale: 图像CFG缩放因子
            timestep_shift: 时间步偏移
            num_timesteps: 去噪步数
            enable_thinking: 是否启用思考模式
            
        Returns:
            PIL.Image: 编辑后的图像
        """
        print(f"开始图像编辑...")
        print(f"输入图像: {input_image_path}")
        print(f"编辑提示: {edit_prompt}")
        
        # 加载输入图像
        try:
            input_image = Image.open(input_image_path).convert('RGB')
            print(f"成功加载图像，尺寸: {input_image.size}")
        except Exception as e:
            raise ValueError(f"无法加载图像 {input_image_path}: {e}")
        
        # 构建输入序列：图像 + 文本提示
        input_sequence = [input_image, edit_prompt]
        
        # 使用interleave_inference进行图像编辑
        print("正在生成编辑后的图像...")
        
        try:
            result = self.inferencer.interleave_inference(
                input_lists=input_sequence,
                think=enable_thinking,
                understanding_output=False,  # 设为False以生成图像
                max_think_token_n=500,
                do_sample=True,
                text_temperature=0.7,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=[0.4, 1.0],
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=0.0,
                cfg_renorm_type="global",
                image_shapes=image_size,
                enable_taylorseer=False,
            )
            
            print(f"生成完成！结果包含 {len(result)} 个元素")
            
            # 查找生成的图像
            generated_image = None
            thinking_text = None
            
            for i, item in enumerate(result):
                if isinstance(item, str):
                    thinking_text = item
                    print(f"思考过程: {item[:200]}...")
                elif isinstance(item, Image.Image):
                    generated_image = item
                    print(f"生成图像 {i}: {item.size}")
            
            if generated_image is None:
                raise ValueError("没有生成图像！")
            
            # 保存结果
            if output_path:
                generated_image.save(output_path)
                print(f"编辑后的图像已保存到: {output_path}")
            
            return generated_image, thinking_text
            
        except Exception as e:
            raise RuntimeError(f"图像编辑过程中出错: {e}")
    
    def test_force_generation(
        self,
        prompt: str,
        image_size: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
    ) -> List[Union[str, Image.Image]]:
        """
        测试强制图像生成功能
        """
        print("🔥 测试强制图像生成")
        
        try:
            result = self.inferencer.test_force_image_generation(
                input_text=prompt,
                image_shapes=image_size,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                num_timesteps=num_timesteps,
            )
            return result
        except Exception as e:
            print(f"强制生成测试失败: {e}")
            return [prompt, f"[错误: {e}]"]
    
    def test_multi_step_generation(
        self,
        instruction: str,
        input_image_path: str = None,
        image_size: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        output_dir: str = "multi_step_outputs",
    ) -> List[Union[str, Image.Image]]:
        """
        测试多轮强制图像编辑功能
        
        这就是你要的功能！基于输入图像，根据复杂指令强制分解为多个步骤，每步编辑生成一张图像
        例如：输入cat.jpg + 指令 "A cat wearing a hat fishing by the water, ink painting style" 
        强制分解编辑：
        - First, the cat wear a hat <image1> (基于原图)
        - Second, the cat fishing by the water <image2> (基于image1)
        - Third, transfer the style to ink painting style <image3> (基于image2)
        """
        print("🚀 测试多轮强制图像编辑")
        print(f"复杂指令: {instruction}")
        if input_image_path:
            print(f"输入图像: {input_image_path}")
        
        try:
            result = self.inferencer.force_multi_step_generation(
                instruction=instruction,
                input_image_path=input_image_path,  # 传递输入图像路径
                image_shapes=image_size,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                num_timesteps=num_timesteps,
                save_intermediate=True,
                output_dir=output_dir,
            )
            
            # 统计生成的图像数量
            image_count = len([x for x in result if hasattr(x, 'size')])
            print(f"✅ 多轮生成完成! 总共生成了 {image_count} 张图像")
            
            return result
            
        except Exception as e:
            print(f"多轮生成失败: {e}")
            import traceback
            traceback.print_exc()
            return [instruction, f"[多轮生成错误: {e}]"]

    def unified_edit_with_text_generation(
        self,
        input_image_path: str, 
        edit_prompt: str,
        max_length: int = 800,
        temperature: float = 0.8,
        image_size: tuple = (1024, 1024),
        force_image_generation: bool = False,
    ) -> List[Union[str, Image.Image]]:
        """
        使用unified_generate进行图像编辑（实验性功能）
        这个方法尝试在一个统一的生成过程中同时处理图像输入和文本输出
        
        Args:
            input_image_path: 输入图像路径
            edit_prompt: 编辑提示词
            max_length: 最大生成长度
            temperature: 温度参数
            image_size: 图像尺寸
            
        Returns:
            包含文本和图像的结果列表
        """
        print(f"使用unified_generate进行图像编辑...")
        
        # 加载图像
        input_image = Image.open(input_image_path).convert('RGB')
        
        # 由于unified_generate目前主要支持文本输入，我们需要先将图像转换为上下文
        # 这里我们使用一个变通的方法
        
        # 构建包含图像信息的提示词
        enhanced_prompt = f"基于提供的图像，{edit_prompt}"
        
        try:
            # 先用图像初始化上下文
            gen_context = self.inferencer.init_gen_context()
            from data.data_utils import pil_img2rgb
            processed_image = self.inferencer.vae_transform.resize_transform(
                pil_img2rgb(input_image)
            )
            gen_context = self.inferencer.update_context_image(
                processed_image, gen_context, vae=True, vit=True
            )
            
            # 然后使用文本生成
            result = self.inferencer.unified_generate(
                input_text=enhanced_prompt,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                image_shapes=image_size,
                cfg_text_scale=4.0,
                cfg_img_scale=1.5,
                return_raw_tokens=False,
                force_image_generation=force_image_generation,
                use_unified_system_prompt=True
            )
            
            return result
            
        except Exception as e:
            print(f"unified编辑方法出错，回退到interleave方法: {e}")
            # 回退到标准的interleave方法
            return self.edit_image_with_prompt(
                input_image_path, edit_prompt, image_size=image_size
            )


def load_bagel_model(model_path="models/BAGEL-7B-MoT", mode=1):
    """
    加载BAGEL模型（复用unified_generation_example.py中的代码）
    
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
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
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
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
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
    """主函数，演示图像编辑功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description="图像编辑功能演示")
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT",
                       help="模型路径")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                       help="加载模式: 1=标准, 2=NF4量化, 3=INT8量化")
    parser.add_argument("--input_image", type=str, 
                       default="/hdd_16T/Zirui/workspace/ATest/Bagel/assets/munchkin-cat-breed.jpg",
                       help="输入图像路径")
    parser.add_argument("--prompt", type=str, 
                       default="A cat wearing a hat fishing by the water, ink painting style",
                       help="编辑提示词")
    parser.add_argument("--output", type=str, default="edited_cat_fishing.png",
                       help="输出图像路径")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024],
                       help="生成图像尺寸 [宽度 高度]")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0,
                       help="文本CFG缩放因子")
    parser.add_argument("--cfg_img_scale", type=float, default=1.5,
                       help="图像CFG缩放因子")
    parser.add_argument("--timesteps", type=int, default=50,
                       help="去噪步数")
    parser.add_argument("--no_thinking", action="store_true",
                       help="禁用思考模式")
    parser.add_argument("--test_force_generation", action="store_true",
                       help="测试强制图像生成功能")
    parser.add_argument("--test_multi_step", action="store_true",
                       help="测试多轮强制图像编辑功能（基于输入图像强制分解为三个编辑步骤）")
    parser.add_argument("--test_unified", action="store_true",
                       help="测试unified_generate方法")
    parser.add_argument("--force_image_gen", action="store_true",
                       help="在unified_generate中强制图像生成")
    parser.add_argument("--output_dir", type=str, default="multi_step_outputs",
                       help="多轮生成的输出目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BAGEL 图像编辑功能演示")
    print("=" * 60)
    print(f"输入图像: {args.input_image}")
    print(f"编辑提示: {args.prompt}")
    print(f"输出路径: {args.output}")
    print(f"图像尺寸: {args.image_size[0]}x{args.image_size[1]}")
    print("=" * 60)
    
    try:
        # 检查输入图像是否存在
        if not os.path.exists(args.input_image):
            raise FileNotFoundError(f"输入图像不存在: {args.input_image}")
        
        # 加载模型
        print("正在加载模型...")
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
            model_path=args.model_path,
            mode=args.mode
        )
        
        # 创建inferencer
        from inferencer import InterleaveInferencer
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        
        # 创建图像编辑demo实例
        demo = ImageEditingDemo(inferencer)
        
        # 根据参数选择测试模式
        if args.test_force_generation:
            print("\n🔥 测试强制图像生成功能...")
            result = demo.test_force_generation(
                prompt=args.prompt,
                image_size=tuple(args.image_size),
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                num_timesteps=args.timesteps,
            )
            
            print(f"强制生成结果包含 {len(result)} 个元素")
            for i, item in enumerate(result):
                if isinstance(item, Image.Image):
                    item.save(args.output)
                    print(f"生成的图像已保存: {args.output}")
                elif isinstance(item, str):
                    print(f"文本内容: {item}")
            return

        elif args.test_multi_step:
            print("\n🚀 测试多轮强制图像编辑功能...")
            print("🎯 这就是你要的功能！基于输入图像强制分解为三个编辑步骤")
            print(f"📝 指令: {args.prompt}")
            print(f"🖼️  输入图像: {args.input_image}")
            print(f"📁 输出目录: {args.output_dir}")
            
            result = demo.test_multi_step_generation(
                instruction=args.prompt,
                input_image_path=args.input_image,  # 传递输入图像路径
                image_size=tuple(args.image_size),
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                num_timesteps=args.timesteps,
                output_dir=args.output_dir,
            )
            
            print(f"\n📊 多轮生成结果包含 {len(result)} 个元素")
            image_count = 0
            for i, item in enumerate(result):
                if isinstance(item, Image.Image):
                    image_count += 1
                    print(f"🖼️  图像 {image_count}: {item.size}")
                elif isinstance(item, str):
                    print(f"📝 步骤: {item}")
            
            print(f"\n🎉 总共编辑生成了 {image_count} 张图像，所有图像已保存到 {args.output_dir}/ 目录")
            print("📋 强制编辑分解步骤:")
            print("  Step 1: Edit this image: First, the cat wear a hat <image1> (基于原图)")
            print("  Step 2: Edit this image: Second, the cat fishing by the water <image2> (基于image1)") 
            print("  Step 3: Edit this image: Third, transfer the style to ink painting style <image3> (基于image2)")
            return

        elif args.test_unified:
            print("\n🧪 测试unified_generate方法...")
            result = demo.unified_edit_with_text_generation(
                input_image_path=args.input_image,
                edit_prompt=args.prompt,
                image_size=tuple(args.image_size),
                force_image_generation=args.force_image_gen,
            )
            
            print(f"Unified生成结果包含 {len(result)} 个元素")
            for i, item in enumerate(result):
                if isinstance(item, Image.Image):
                    item.save(args.output)
                    print(f"生成的图像已保存: {args.output}")
                elif isinstance(item, str):
                    print(f"文本内容: {item}")
            return
        
        else:
            # 标准图像编辑流程
            print("\n开始图像编辑...")
            edited_image, thinking = demo.edit_image_with_prompt(
                input_image_path=args.input_image,
                edit_prompt=args.prompt,
                output_path=args.output,
                image_size=tuple(args.image_size),
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                num_timesteps=args.timesteps,
                enable_thinking=not args.no_thinking
            )
        
        print("\n" + "=" * 60)
        print("图像编辑完成！")
        print(f"编辑后的图像已保存为: {args.output}")
        print(f"图像尺寸: {edited_image.size}")
        
        if thinking and not args.no_thinking:
            print("\n模型思考过程:")
            print(thinking)
        
        print("=" * 60)
        
    except Exception as e:
        print(f"图像编辑过程中出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保:")
        print("1. 模型路径正确且模型文件完整")
        print("2. 输入图像文件存在且可读取")
        print("3. 有足够的GPU显存")
        print("4. 所有依赖库已正确安装")


if __name__ == "__main__":
    main()
