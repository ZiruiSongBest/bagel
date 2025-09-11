#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自回归交错生成模型推理脚本

该脚本基于您训练的统一FSDP模型检查点，实现文本和图像的交错自回归生成。
支持多种生成模式：文本->图像、图像->文本、混合生成等。

使用方式:
    python inference_unified_autoregressive.py \
        --checkpoint_path /workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800 \
        --prompt "描述一只猫在花园里玩耍，然后画出这个场景。" \
        --output_dir outputs
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from PIL import Image
import json
import logging
from datetime import datetime
from copy import deepcopy

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from inferencer import InterleaveInferencer
from train.fsdp_utils import FSDPCheckpoint

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedModelLoader:
    """统一模型加载器，支持从FSDP训练检查点加载模型"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Args:
            checkpoint_path: FSDP训练检查点路径
            device: 推理设备
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None
        self.vae_model = None
        self.tokenizer = None
        self.new_token_ids = None
        self.config = None
        
        # 验证检查点路径
        if not self.checkpoint_path.exists():
            raise ValueError(f"检查点路径不存在: {checkpoint_path}")
        
        # 检查必要的文件
        required_files = ["model.safetensors", "ema.safetensors"]
        for file in required_files:
            if not (self.checkpoint_path / file).exists():
                logger.warning(f"检查点文件缺失: {file}")
    
    def load_model_config(self) -> Dict[str, Any]:
        """加载模型配置，通过分析检查点和推断配置"""
        logger.info("正在推断模型配置...")
        
        # 默认配置（基于训练脚本中的设置）
        config = {
            # LLM配置
            "llm_path": "models/Qwen2.5-0.5B-Instruct",
            "layer_module": "Qwen2MoTDecoderLayer",
            "llm_qk_norm": True,
            "tie_word_embeddings": False,
            
            # VAE配置
            "vae_path": "models/flux/ae.safetensors",
            "latent_patch_size": 2,
            "max_latent_size": 64,  # 从训练脚本推断
            
            # ViT配置
            "vit_path": "models/siglip-so400m-14-980-flash-attn2-navit/",
            "vit_patch_size": 14,
            "vit_max_num_patch_per_side": 70,
            "vit_select_layer": -2,
            "vit_rope": False,
            
            # 其他配置
            "connector_act": "gelu_pytorch_tanh",
            "interpolate_pos": False,
            "timestep_shift": 1.0,
            "visual_gen": True,
            "visual_und": True,
        }
        
        # 尝试从相对路径查找模型文件
        base_paths = [
            self.checkpoint_path.parent.parent.parent,  # results/training/checkpoints -> bagel
            Path("/workspace/bagel"),  # 绝对路径
            Path.cwd(),  # 当前目录
        ]
        
        for base_path in base_paths:
            if (base_path / "models").exists():
                config["base_path"] = str(base_path)
                for key in ["llm_path", "vae_path", "vit_path"]:
                    if not config[key].startswith("/"):
                        config[key] = str(base_path / config[key])
                break
        else:
            logger.warning("未找到模型文件基础路径，使用相对路径")
            config["base_path"] = str(Path.cwd())
        
        logger.info(f"配置模型基础路径: {config.get('base_path', 'N/A')}")
        return config
    
    def load_components(self, config: Dict[str, Any]):
        """加载模型组件"""
        logger.info("正在加载模型组件...")
        
        # 1. 加载LLM配置和模型
        logger.info("加载语言模型...")
        if os.path.exists(config["llm_path"]):
            llm_config = Qwen2Config.from_pretrained(config["llm_path"])
        else:
            logger.warning(f"LLM路径不存在: {config['llm_path']}, 使用默认配置")
            llm_config = Qwen2Config()
        
        llm_config.layer_module = config["layer_module"]
        llm_config.qk_norm = config["llm_qk_norm"]
        llm_config.tie_word_embeddings = config["tie_word_embeddings"]
        
        if os.path.exists(config["llm_path"]):
            language_model = Qwen2ForCausalLM.from_pretrained(config["llm_path"], config=llm_config)
        else:
            language_model = Qwen2ForCausalLM(llm_config)
        
        # 2. 加载ViT模型（如果启用视觉理解）
        vit_model = None
        if config["visual_und"]:
            logger.info("加载视觉理解模型...")
            if os.path.exists(config["vit_path"]):
                vit_config = SiglipVisionConfig.from_pretrained(config["vit_path"])
            else:
                logger.warning(f"ViT路径不存在: {config['vit_path']}, 使用默认配置")
                vit_config = SiglipVisionConfig()
            
            vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + config["vit_select_layer"]
            vit_config.rope = config["vit_rope"]
            
            if os.path.exists(config["vit_path"]):
                vit_model = SiglipVisionModel.from_pretrained(config["vit_path"], config=vit_config)
            else:
                vit_model = SiglipVisionModel(vit_config)
        
        # 3. 加载VAE模型（如果启用视觉生成）
        vae_model = None
        vae_config = None
        if config["visual_gen"]:
            logger.info("加载VAE模型...")
            if os.path.exists(config["vae_path"]):
                vae_model, vae_config = load_ae(config["vae_path"])
            else:
                logger.error(f"VAE路径不存在: {config['vae_path']}")
                raise FileNotFoundError(f"VAE文件不存在: {config['vae_path']}")
        
        # 4. 创建Bagel配置
        bagel_config = BagelConfig(
            visual_gen=config["visual_gen"],
            visual_und=config["visual_und"],
            llm_config=llm_config,
            vit_config=vit_config if config["visual_und"] else None,
            vae_config=vae_config if config["visual_gen"] else None,
            latent_patch_size=config["latent_patch_size"],
            max_latent_size=config["max_latent_size"],
            vit_max_num_patch_per_side=config["vit_max_num_patch_per_side"],
            connector_act=config["connector_act"],
            interpolate_pos=config["interpolate_pos"],
            timestep_shift=config["timestep_shift"],
        )
        
        # 5. 创建Bagel模型
        logger.info("创建Bagel模型...")
        model = Bagel(language_model, vit_model, bagel_config)
        
        # 6. 处理ViT模型的特殊配置
        if config["visual_und"]:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        
        # 7. 加载tokenizer
        logger.info("加载分词器...")
        if os.path.exists(config["llm_path"]):
            tokenizer = Qwen2Tokenizer.from_pretrained(config["llm_path"])
        else:
            logger.warning("使用默认分词器")
            tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        
        if num_new_tokens > 0:
            logger.info(f"调整词汇表大小，新增{num_new_tokens}个token")
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)
        
        return model, vae_model, tokenizer, new_token_ids, bagel_config
    
    def load_checkpoint_weights(self, model):
        """从FSDP检查点加载权重"""
        logger.info(f"正在从检查点加载权重: {self.checkpoint_path}")
        
        # 首先尝试加载EMA权重（通常质量更好）
        ema_path = self.checkpoint_path / "ema.safetensors"
        model_path = self.checkpoint_path / "model.safetensors"
        
        if ema_path.exists():
            logger.info("使用EMA权重")
            checkpoint_path = ema_path
        elif model_path.exists():
            logger.info("使用普通模型权重")
            checkpoint_path = model_path
        else:
            raise FileNotFoundError(f"找不到模型权重文件: {ema_path} 或 {model_path}")
        
        # 加载权重（使用safetensors格式）
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            logger.info(f"成功加载权重，包含{len(state_dict)}个参数张量")
            
            # 加载到模型
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"缺失的键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                logger.warning(f"意外的键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            
            logger.info("模型权重加载完成")
            
        except Exception as e:
            logger.error(f"加载权重失败: {e}")
            raise
    
    def load_model(self) -> tuple:
        """完整的模型加载流程"""
        logger.info("开始加载统一生成模型...")
        
        # 1. 加载配置
        config = self.load_model_config()
        self.config = config
        
        # 2. 加载组件
        model, vae_model, tokenizer, new_token_ids, bagel_config = self.load_components(config)
        
        # 3. 加载检查点权重
        self.load_checkpoint_weights(model)
        
        # 4. 移动到指定设备并设置为评估模式
        model = model.to(self.device)
        model.eval()
        
        if vae_model is not None:
            vae_model = vae_model.to(self.device)
            vae_model.eval()
        
        # 5. 存储加载的组件
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        
        logger.info("模型加载完成，准备进行推理")
        
        return model, vae_model, tokenizer, new_token_ids, bagel_config


class UnifiedInferenceEngine:
    """统一推理引擎，支持多种生成模式"""
    
    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        new_token_ids,
        device: str = "cuda"
    ):
        """
        Args:
            model: Bagel模型
            vae_model: VAE模型
            tokenizer: 分词器
            new_token_ids: 特殊token ID映射
            device: 推理设备
        """
        self.device = device
        
        # 创建图像变换器
        self.vae_transform = ImageTransform(1024, 512, 16)  # VAE变换
        self.vit_transform = ImageTransform(980, 224, 14)   # ViT变换
        
        # 创建InterleaveInferencer
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=new_token_ids
        )
        
        logger.info("推理引擎初始化完成")
    
    def autoregressive_generate(
        self,
        prompt: str,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        image_shapes: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        save_intermediate: bool = False,
        output_dir: str = "outputs"
    ) -> List[Union[str, Image.Image]]:
        """
        自回归交错生成：根据文本prompt自动决定何时生成文本和图像
        
        Args:
            prompt: 输入文本提示
            max_length: 最大生成长度
            do_sample: 是否采样生成
            temperature: 采样温度
            image_shapes: 生成图像的尺寸
            cfg_text_scale: 文本CFG比例
            cfg_img_scale: 图像CFG比例
            num_timesteps: 扩散步数
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录
            
        Returns:
            包含文本和图像的混合列表
        """
        logger.info(f"开始自回归交错生成，输入: {prompt[:100]}...")
        
        # 创建输出目录
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
        
        # 使用统一生成方法
        result = self.inferencer.unified_generate(
            input_text=prompt,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            image_shapes=image_shapes,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            num_timesteps=num_timesteps,
            use_unified_system_prompt=True
        )
        
        # 保存结果
        if save_intermediate:
            self._save_generation_results(result, output_dir)
        
        return result
    
    def step_by_step_generate(
        self,
        instructions: List[str],
        initial_image: Optional[Image.Image] = None,
        image_shapes: tuple = (1024, 1024),
        **generation_kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        分步生成：按照指令列表逐步生成内容
        
        Args:
            instructions: 指令列表，每个指令可能生成文本或图像
            initial_image: 初始图像（可选）
            image_shapes: 图像尺寸
            **generation_kwargs: 其他生成参数
            
        Returns:
            生成结果列表
        """
        logger.info(f"开始分步生成，共{len(instructions)}个指令")
        
        all_results = []
        current_context = []
        
        # 如果有初始图像，添加到上下文
        if initial_image is not None:
            current_context.append(initial_image)
            all_results.append(initial_image)
        
        for i, instruction in enumerate(instructions):
            logger.info(f"执行步骤 {i+1}/{len(instructions)}: {instruction[:50]}...")
            
            try:
                # 构建当前步骤的输入
                step_input = current_context + [instruction]
                
                # 使用交错推理
                step_result = self.inferencer.interleave_inference(
                    input_lists=step_input,
                    image_shapes=image_shapes,
                    **generation_kwargs
                )
                
                # 更新上下文和结果
                current_context.extend([instruction])
                all_results.extend([f"步骤 {i+1}: {instruction}"])
                
                for item in step_result:
                    all_results.append(item)
                    if isinstance(item, Image.Image):
                        current_context.append(item)
                
                logger.info(f"步骤 {i+1} 完成，生成了 {len(step_result)} 个项目")
                
            except Exception as e:
                logger.error(f"步骤 {i+1} 失败: {e}")
                all_results.append(f"[步骤 {i+1} 失败: {e}]")
        
        return all_results
    
    def image_editing_chain(
        self,
        input_image: Image.Image,
        edit_instructions: List[str],
        image_shapes: tuple = (1024, 1024),
        **generation_kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        图像编辑链：对图像进行连续编辑
        
        Args:
            input_image: 输入图像
            edit_instructions: 编辑指令列表
            image_shapes: 图像尺寸
            **generation_kwargs: 其他生成参数
            
        Returns:
            编辑历程和结果图像
        """
        logger.info(f"开始图像编辑链，共{len(edit_instructions)}步编辑")
        
        results = [input_image]
        current_image = input_image
        
        for i, instruction in enumerate(edit_instructions):
            logger.info(f"编辑步骤 {i+1}: {instruction[:50]}...")
            
            try:
                # 构建编辑提示
                edit_prompt = f"Edit this image: {instruction}"
                
                # 使用交错推理进行编辑
                edit_result = self.inferencer.interleave_inference(
                    input_lists=[current_image, edit_prompt],
                    understanding_output=False,  # 生成图像而不是理解
                    image_shapes=image_shapes,
                    **generation_kwargs
                )
                
                # 更新当前图像
                for item in edit_result:
                    if isinstance(item, Image.Image):
                        current_image = item
                        results.append(f"编辑 {i+1}: {instruction}")
                        results.append(item)
                        break
                else:
                    logger.warning(f"编辑步骤 {i+1} 未生成图像")
                    results.append(f"[编辑 {i+1} 失败: 未生成图像]")
                
            except Exception as e:
                logger.error(f"编辑步骤 {i+1} 失败: {e}")
                results.append(f"[编辑 {i+1} 失败: {e}]")
        
        return results
    
    def _save_generation_results(self, results: List[Union[str, Image.Image]], output_dir: str):
        """保存生成结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存文本内容
        text_content = []
        image_info = []
        
        for i, item in enumerate(results):
            if isinstance(item, str):
                text_content.append(f"[{i}] {item}")
            elif isinstance(item, Image.Image):
                image_filename = f"generated_image_{i}_{timestamp}.png"
                image_path = os.path.join(output_dir, image_filename)
                item.save(image_path)
                image_info.append({"index": i, "filename": image_filename, "size": item.size})
                text_content.append(f"[{i}] [IMAGE: {image_filename}]")
        
        # 保存文本摘要
        text_file = os.path.join(output_dir, f"generation_summary_{timestamp}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=== 自回归交错生成结果 ===\n\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"总项目数: {len(results)}\n")
            f.write(f"图像数量: {len(image_info)}\n\n")
            f.write("=== 详细内容 ===\n")
            for line in text_content:
                f.write(line + "\n")
        
        # 保存元数据
        metadata = {
            "timestamp": timestamp,
            "total_items": len(results),
            "image_count": len(image_info),
            "images": image_info
        }
        
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_dir}")


def create_inference_engine(checkpoint_path: str, device: str = "cuda") -> UnifiedInferenceEngine:
    """创建推理引擎的便捷函数"""
    
    # 加载模型
    loader = UnifiedModelLoader(checkpoint_path, device)
    model, vae_model, tokenizer, new_token_ids, config = loader.load_model()
    
    # 创建推理引擎
    engine = UnifiedInferenceEngine(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        device=device
    )
    
    return engine


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description="自回归交错生成模型推理")
    
    # 模型和输入参数
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="FSDP训练检查点路径")
    parser.add_argument("--prompt", type=str, required=True,
                       help="输入文本提示")
    parser.add_argument("--input_image", type=str, default=None,
                       help="输入图像路径（可选）")
    
    # 生成参数
    parser.add_argument("--max_length", type=int, default=500,
                       help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="采样温度")
    parser.add_argument("--do_sample", action="store_true",
                       help="启用采样生成")
    parser.add_argument("--image_width", type=int, default=1024,
                       help="生成图像宽度")
    parser.add_argument("--image_height", type=int, default=1024,
                       help="生成图像高度")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0,
                       help="文本CFG比例")
    parser.add_argument("--cfg_img_scale", type=float, default=1.5,
                       help="图像CFG比例")
    parser.add_argument("--num_timesteps", type=int, default=50,
                       help="扩散步数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="输出目录")
    parser.add_argument("--save_intermediate", action="store_true",
                       help="保存中间结果")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                       help="推理设备")
    
    # 模式参数
    parser.add_argument("--mode", type=str, default="autoregressive",
                       choices=["autoregressive", "step_by_step", "image_editing"],
                       help="生成模式")
    
    args = parser.parse_args()
    
    try:
        # 创建推理引擎
        logger.info("正在初始化推理引擎...")
        engine = create_inference_engine(args.checkpoint_path, args.device)
        
        # 准备输入
        input_image = None
        if args.input_image:
            if os.path.exists(args.input_image):
                input_image = Image.open(args.input_image).convert('RGB')
                logger.info(f"已加载输入图像: {input_image.size}")
            else:
                logger.warning(f"输入图像文件不存在: {args.input_image}")
        
        # 执行推理
        image_shapes = (args.image_height, args.image_width)
        generation_kwargs = {
            "max_length": args.max_length,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
            "num_timesteps": args.num_timesteps,
            "save_intermediate": args.save_intermediate,
            "output_dir": args.output_dir
        }
        
        if args.mode == "autoregressive":
            logger.info("使用自回归生成模式")
            results = engine.autoregressive_generate(
                prompt=args.prompt,
                image_shapes=image_shapes,
                **generation_kwargs
            )
            
        elif args.mode == "step_by_step":
            logger.info("使用分步生成模式")
            # 将提示按句子分割
            instructions = [s.strip() for s in args.prompt.split('.') if s.strip()]
            results = engine.step_by_step_generate(
                instructions=instructions,
                initial_image=input_image,
                image_shapes=image_shapes,
                **{k: v for k, v in generation_kwargs.items() if k != "prompt"}
            )
            
        elif args.mode == "image_editing":
            if input_image is None:
                raise ValueError("图像编辑模式需要提供输入图像")
            logger.info("使用图像编辑模式")
            edit_instructions = [s.strip() for s in args.prompt.split('.') if s.strip()]
            results = engine.image_editing_chain(
                input_image=input_image,
                edit_instructions=edit_instructions,
                image_shapes=image_shapes,
                **{k: v for k, v in generation_kwargs.items() if k not in ["prompt", "max_length"]}
            )
        
        # 显示结果摘要
        text_count = sum(1 for item in results if isinstance(item, str))
        image_count = sum(1 for item in results if isinstance(item, Image.Image))
        
        logger.info(f"生成完成！总计 {len(results)} 项 (文本: {text_count}, 图像: {image_count})")
        
        # 简单展示前几项结果
        for i, item in enumerate(results[:5]):
            if isinstance(item, str):
                logger.info(f"[{i}] 文本: {item[:100]}{'...' if len(item) > 100 else ''}")
            elif isinstance(item, Image.Image):
                logger.info(f"[{i}] 图像: {item.size}")
        
        if len(results) > 5:
            logger.info(f"... 还有 {len(results) - 5} 项结果")
        
    except Exception as e:
        logger.error(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
