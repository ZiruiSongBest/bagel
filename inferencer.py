# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> <|vision_start|> image will be generated here <|vision_end|>

When you want to generate an image, use the special tokens <|vision_start|> and <|vision_end|> to indicate where the image should be placed.'''

# 新增：专门用于统一生成的系统提示词
UNIFIED_GEN_SYSTEM_PROMPT = '''You are a multimodal AI that can generate both text and images. When generating content:

1. For text: Simply generate text naturally
2. For images: Use the special tokens <|vision_start|> and <|vision_end|> to mark where an image should be generated

Example format:
- Text response: "This is a text response."
- Image generation: "Here is your image: <|vision_start|> <|vision_end|>"
- Mixed response: "I'll create an image for you: <|vision_start|> <|vision_end|> And here's some additional text."

Always use <|vision_start|> and <|vision_end|> tokens when you want to generate an image.'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        # 用于在统一生成过程中存储生成的图像
        self._generated_images = {}
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            device = next(self.model.parameters()).device
            generation_input = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in generation_input.items()
            }
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            device = next(self.model.parameters()).device
            generation_input = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in generation_input.items()
            }
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        return_tokens: bool = False,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        if return_tokens:
            generated_tokens = unpacked_latent[:, 0].tolist()
            return output, generated_tokens

        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        return output_list
    
    def _setup_generation_step(self, gen_context, new_token_id=None):
        """设置生成步骤的输入参数"""
        if new_token_id is None:
            new_token_id = self.new_token_ids.get('bos_token_id', 1)
            
        curr_tokens = torch.tensor([new_token_id], dtype=torch.long, device=self.model.device)
        packed_text_embedding = self.model.language_model.model.embed_tokens(curr_tokens)
        query_lens = torch.ones_like(curr_tokens)
        
        kv_lens = torch.tensor(gen_context['kv_lens'], dtype=torch.int, device=self.model.device)
        ropes = torch.tensor(gen_context['ropes'], dtype=torch.long, device=self.model.device)
        
        packed_query_indexes = torch.cumsum(kv_lens, dim=0) + torch.arange(
            0, len(kv_lens), 
            device=kv_lens.device, 
            dtype=kv_lens.dtype
        )
        
        # 构建packed_key_value_indexes - 这是关键部分
        uppacked = []
        start_idx = 0
        for i, kv_len in enumerate(kv_lens.tolist()):
            uppacked.append(torch.arange(kv_len, device=kv_lens.device, dtype=torch.long) + i)
        packed_key_value_indexes = torch.cat(uppacked, dim=0)
        
        # 构建position_ids
        packed_query_position_ids = ropes
        
        return {
            'packed_text_embedding': packed_text_embedding,
            'query_lens': query_lens, 
            'packed_query_position_ids': packed_query_position_ids,
            'packed_query_indexes': packed_query_indexes,
            'past_key_values': gen_context['past_key_values'],
            'key_values_lens': kv_lens,
            'packed_key_value_indexes': packed_key_value_indexes,
        }
    
    def _predict_next_token_logits(self, gen_context, new_token_id=None):
        """预测下一个token的logits"""
        generation_input = self._setup_generation_step(gen_context, new_token_id)
        
        extra_inputs = {}
        if self.model.use_moe:
            extra_inputs = {"mode": "und"}
            
        output = self.model.language_model.forward_inference(
            packed_query_sequence=generation_input['packed_text_embedding'],
            query_lens=generation_input['query_lens'],
            packed_query_position_ids=generation_input['packed_query_position_ids'],
            packed_query_indexes=generation_input['packed_query_indexes'],
            past_key_values=generation_input['past_key_values'],
            key_values_lens=generation_input['key_values_lens'],
            packed_key_value_indexes=generation_input['packed_key_value_indexes'],
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        
        pred_logits = self.model.language_model.lm_head(output.packed_query_sequence)
        
        # 更新gen_context
        updated_gen_context = {
            'past_key_values': output.past_key_values,
            'kv_lens': [kv_len + 1 for kv_len in gen_context['kv_lens']],
            'ropes': [rope + 1 for rope in gen_context['ropes']]
        }
        
        return pred_logits, updated_gen_context
    
    def _predict_next_token_logits_with_input(self, gen_context, generation_input):
        """使用预定义的generation_input预测下一个token的logits"""
        extra_inputs = {}
        if self.model.use_moe:
            extra_inputs = {"mode": "und"}
            
        # 准备输入
        packed_start_tokens = generation_input['packed_start_tokens']
        packed_text_embedding = self.model.language_model.model.embed_tokens(packed_start_tokens)
        query_lens = torch.ones_like(packed_start_tokens)
        
        # 计算packed_query_indexes
        kv_lens = generation_input['key_values_lens']
        packed_query_indexes = torch.cumsum(kv_lens, dim=0) + torch.arange(
            0, len(kv_lens), 
            device=kv_lens.device, 
            dtype=kv_lens.dtype
        )
        
        output = self.model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=generation_input['packed_query_position_ids'],
            packed_query_indexes=packed_query_indexes,
            past_key_values=gen_context['past_key_values'],
            key_values_lens=generation_input['key_values_lens'],
            packed_key_value_indexes=generation_input['packed_key_value_indexes'],
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        
        pred_logits = self.model.language_model.lm_head(output.packed_query_sequence)
        
        # 更新gen_context
        updated_gen_context = {
            'past_key_values': output.past_key_values,
            'kv_lens': [kv_len + 1 for kv_len in gen_context['kv_lens']],
            'ropes': [rope + 1 for rope in gen_context['ropes']]
        }
        
        return pred_logits, updated_gen_context

    def _update_cache_with_token(self, token_id, gen_context):
        """使用单个token更新缓存"""
        try:
            # 为单个token构造输入，检查token_id是否有效
            if token_id is None or not isinstance(token_id, int):
                print(f"警告: 无效的token_id: {token_id}")
                return gen_context
                
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            
            # 检查解码结果是否有效
            if token_text is None or token_text == "":
                print(f"警告: token_id {token_id} 解码为空或None")
                return gen_context
            
            # 使用现有的update_context_text方法，但需要确保它能处理单个token
            temp_context = deepcopy(gen_context)
            updated_context = self.update_context_text(token_text, temp_context)
            
            return updated_context
            
        except Exception as e:
            print(f"警告: 更新token缓存时出错 (token_id={token_id}): {e}")
            return gen_context
    
    @torch.no_grad()
    def unified_generate(
        self,
        input_text: str,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        image_shapes: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: tuple = (0.4, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        enable_taylorseer: bool = False,
        return_raw_tokens: bool = False,
        force_image_generation: bool = False,
        use_unified_system_prompt: bool = True,
    ) -> Union[List[Union[str, Image.Image]], List[int]]:
        """统一的多模态自回归生成方法
        
        使用一个简化的方法：利用现有的gen_text生成，但监控特殊token
        """
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # 初始化生成上下文
            gen_context = self.init_gen_context()
            
            # 添加系统提示词（如果启用）
            if use_unified_system_prompt:
                gen_context = self.update_context_text(UNIFIED_GEN_SYSTEM_PROMPT, gen_context)
            
            # 如果强制生成图像，修改输入文本以包含图像token
            if force_image_generation and '<|vision_start|>' not in input_text:
                input_text = f"{input_text} <|vision_start|> <|vision_end|>"
                
            gen_context = self.update_context_text(input_text, gen_context)
            
            # 保存CFG上下文快照
            cfg_text_context_snapshot = deepcopy(gen_context)
            cfg_img_context_snapshot = deepcopy(gen_context)
            
            # 使用改进的文本生成方法
            result = self._unified_generate_with_monitoring(
                gen_context=gen_context,
                cfg_text_context=cfg_text_context_snapshot,
                cfg_img_context=cfg_img_context_snapshot,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                image_shapes=image_shapes,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
                return_raw_tokens=return_raw_tokens
            )
            
            return result
    
    def _unified_generate_with_monitoring(
        self, 
        gen_context,
        cfg_text_context,
        cfg_img_context,
        max_length,
        do_sample,
        temperature,
        image_shapes,
        cfg_text_scale,
        cfg_img_scale,
        cfg_interval,
        timestep_shift,
        num_timesteps,
        cfg_renorm_min,
        cfg_renorm_type,
        enable_taylorseer,
        return_raw_tokens
    ):
        """使用现有generate_text逻辑但监控图像token的生成方法"""
        
        # 准备生成输入，基于现有的gen_text方法
        generation_input = self.model.prepare_start_tokens(
            gen_context['kv_lens'], 
            gen_context['ropes'], 
            self.new_token_ids
        )
        
        # 使用类似于model.generate_text的逻辑，但加入图像生成检测
        past_key_values = gen_context['past_key_values']
        generated_sequence = []
        generated_images = []
        curr_tokens = generation_input['packed_start_tokens']
        step = 0
        
        while step < max_length:
            generated_sequence.append(curr_tokens)
            
            # 文本嵌入
            packed_text_embedding = self.model.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            
            kv_lens = generation_input['key_values_lens']
            packed_query_indexes = torch.cumsum(kv_lens, dim=0) + torch.arange(
                0, len(kv_lens), device=kv_lens.device, dtype=kv_lens.dtype
            )
            
            # 更新packed_key_value_indexes
            packed_key_value_indexes = generation_input['packed_key_value_indexes']
            uppacked = list(packed_key_value_indexes.split(kv_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            
            extra_inputs = {}
            if self.model.use_moe:
                extra_inputs = {"mode": "und"}
                
            output = self.model.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=generation_input['packed_query_position_ids'],
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=kv_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.model.language_model.lm_head(packed_query_sequence)
            
            # 选择下一个token
            if do_sample:
                import torch.nn.functional as F
                probs = F.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)
            
            # 检查是否是图像开始token
            next_token_id = curr_tokens[0].item()
            
            if next_token_id == self.new_token_ids.get('start_of_image'):
                # 触发图像生成
                try:
                    # 重建gen_context用于图像生成
                    current_gen_context = {
                        'past_key_values': past_key_values,
                        'kv_lens': kv_lens.tolist(),
                        'ropes': generation_input['packed_query_position_ids'].tolist()
                    }
                    
                    img = self.gen_image(
                        image_shapes,
                        current_gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=cfg_img_context,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                        enable_taylorseer=enable_taylorseer,
                    )
                    
                    generated_images.append(img)
                    self._generated_images[f"img_{step}"] = img
                    
                    # 更新上下文以包含图像
                    current_gen_context = self.update_context_image(img, current_gen_context, vae=True, vit=True)
                    past_key_values = current_gen_context['past_key_values']
                    
                    # 添加图像结束token
                    if 'end_of_image' in self.new_token_ids:
                        generated_sequence.append(torch.tensor([self.new_token_ids['end_of_image']], device=curr_tokens.device))
                        
                except Exception as e:
                    print(f"图像生成失败: {e}")
            
            # 更新生成参数
            uppacked = list(packed_key_value_indexes.split(kv_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat([
                    uppacked[i], 
                    torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)
                ], dim=0)
            
            generation_input['packed_key_value_indexes'] = torch.cat(uppacked, dim=0)
            generation_input['key_values_lens'] = kv_lens + 1
            generation_input['packed_query_position_ids'] = generation_input['packed_query_position_ids'] + 1
            step += 1
            
            # 检查结束条件
            if next_token_id == self.new_token_ids.get('eos_token_id'):
                break
        
        # 组装输出
        if return_raw_tokens:
            # 返回原始token序列
            all_tokens = []
            for seq in generated_sequence:
                all_tokens.extend(seq.tolist())
            return all_tokens
        else:
            # 解码文本并组合图像
            output_sequence = []
            
            # 解码所有生成的token
            all_tokens = []
            for seq in generated_sequence:
                all_tokens.extend(seq.tolist())
            
            # 简单版本：生成文本 + 图像
            if all_tokens:
                try:
                    text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
                    if text.strip():
                        output_sequence.append(text)
                except:
                    output_sequence.append("[文本解码失败]")
            
            # 添加生成的图像
            for img in generated_images:
                output_sequence.append(img)
                
            return output_sequence
    
    @torch.no_grad() 
    def test_force_image_generation(
        self,
        input_text: str,
        image_shapes: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """测试方法：强制触发图像生成
        
        这个方法会在输入文本后直接添加图像token，强制触发图像生成过程
        """
        print("🔥 强制图像生成测试模式")
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # 初始化上下文
            gen_context = self.init_gen_context()
            gen_context = self.update_context_text(input_text, gen_context)
            
            # 保存CFG上下文
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)
            
            # 强制添加图像开始token
            start_token_id = self.new_token_ids.get('start_of_image')
            if start_token_id is None:
                raise ValueError("找不到图像开始token ID")
                
            print(f"图像开始token ID: {start_token_id}")
            
            # 手动触发图像生成
            try:
                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    num_timesteps=num_timesteps,
                    **kwargs
                )
                
                return [input_text, img]
                
            except Exception as e:
                print(f"强制图像生成失败: {e}")
                return [input_text, f"[图像生成失败: {e}]"]
    
    @torch.no_grad()
    def force_multi_step_generation(
        self,
        instruction: str,
        input_image_path: str = None,
        image_shapes: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        num_timesteps: int = 50,
        save_intermediate: bool = True,
        output_dir: str = "multi_step_outputs",
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """多轮强制图像编辑生成
        
        根据复杂指令强制分解为多个步骤，每个步骤基于前一步的图像进行编辑
        例如：
        输入图像: cat.jpg + 指令: "A cat wearing a hat fishing by the water, ink painting style"
        强制分解: 
        - First, the cat wear a hat <image1> (基于原图)
        - Second, the cat fishing by the water <image2> (基于image1)
        - Third, transfer the style to ink painting style <image3> (基于image2)
        """
        print("🚀 多轮强制图像编辑模式")
        print(f"原始指令: {instruction}")
        if input_image_path:
            print(f"输入图像: {input_image_path}")
        
        # 清空之前的生成图像缓存
        self._generated_images.clear()
        
        # 创建输出目录
        if save_intermediate:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        # 加载初始图像（如果提供）
        initial_image = None
        if input_image_path:
            try:
                initial_image = Image.open(input_image_path).convert('RGB')
                print(f"✅ 成功加载初始图像: {initial_image.size}")
            except Exception as e:
                print(f"❌ 加载初始图像失败: {e}")
                return [f"[图像加载失败: {e}]"]
        
        # 写死的三步骤分解（基于图像编辑的测试用）
        forced_steps = [
            "First, the cat wear a hat",
            "Second, the cat fishing by the water", 
            "Third, transfer the style to ink painting style"
        ]
        
        print(f"🎯 强制分解为 {len(forced_steps)} 个步骤:")
        for i, step in enumerate(forced_steps, 1):
            print(f"  步骤 {i}: {step}")
        
        output_sequence = []
        current_image = initial_image  # 从初始图像开始
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for step_idx, step_description in enumerate(forced_steps):
                print(f"\n🎯 执行步骤 {step_idx + 1}: {step_description}")
                
                try:
                    # 初始化上下文
                    gen_context = self.init_gen_context()
                    
                    # 每一步都基于当前图像（如果存在）
                    if current_image is not None:
                        processed_image = self.vae_transform.resize_transform(
                            pil_img2rgb(current_image)
                        )
                        gen_context = self.update_context_image(
                            processed_image, gen_context, vae=True, vit=True
                        )
                        print(f"📸 基于当前图像进行编辑，尺寸: {current_image.size}")
                    
                    # 构建步骤提示词
                    if step_idx == 0:
                        # 第一步：基于原图，添加帽子
                        step_prompt = step_description if current_image is None else f"Edit this image: {step_description}"
                    else:
                        # 后续步骤：基于前一步结果继续编辑
                        step_prompt = f"Edit this image: {step_description}"
                    
                    gen_context = self.update_context_text(step_prompt, gen_context)
                    
                    # 保存CFG上下文
                    cfg_text_context = deepcopy(gen_context)
                    cfg_img_context = deepcopy(gen_context)
                    
                    # 生成图像
                    img = self.gen_image(
                        image_shapes,
                        gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=cfg_img_context,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        num_timesteps=num_timesteps,
                        **kwargs
                    )
                    
                    # 保存到缓存
                    img_id = f"image{step_idx + 1}"
                    self._generated_images[img_id] = img
                    current_image = img
                    
                    # 添加到输出序列
                    output_sequence.append(f"{step_description} <image{step_idx + 1}>")
                    output_sequence.append(img)
                    
                    # 保存中间结果
                    if save_intermediate:
                        img_path = os.path.join(output_dir, f"step_{step_idx + 1}_{img_id}.png")
                        img.save(img_path)
                        print(f"💾 已保存: {img_path}")
                    
                    print(f"✅ 步骤 {step_idx + 1} 完成，生成图像尺寸: {img.size}")
                    
                except Exception as e:
                    error_msg = f"❌ 步骤 {step_idx + 1} 失败: {e}"
                    print(error_msg)
                    output_sequence.append(error_msg)
                    import traceback
                    traceback.print_exc()
                    # 继续执行下一步骤
                    continue
        
        image_count = len([x for x in output_sequence if hasattr(x, 'size')])
        print(f"\n🎉 多轮生成完成! 生成了 {image_count} 张图像")
        return output_sequence
    
    def _parse_token_sequence(self, token_sequence: List[int]) -> List[Union[str, Image.Image]]:
        """解析token序列，提取文本片段和实际生成的图像"""
        output_sequence = []
        text_buffer = []
        image_counter = 0
        i = 0
        
        while i < len(token_sequence):
            token_id = token_sequence[i]
            
            if token_id == self.new_token_ids.get('start_of_image'):
                # 遇到图像开始token
                if text_buffer:
                    # 先添加之前的文本
                    try:
                        # 过滤掉None值
                        valid_tokens = [t for t in text_buffer if t is not None and isinstance(t, int)]
                        if valid_tokens:
                            text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                            if text and text.strip():
                                output_sequence.append(text)
                    except Exception as e:
                        print(f"警告: 解码图像前文本时出错: {e}")
                    text_buffer = []
                
                # 查找对应的图像结束token
                j = i + 1
                while j < len(token_sequence) and token_sequence[j] != self.new_token_ids.get('end_of_image'):
                    j += 1
                
                # 从存储中获取实际生成的图像
                img_id = f"img_{i}"  # 使用token位置作为图像ID
                if img_id in self._generated_images:
                    output_sequence.append(self._generated_images[img_id])
                else:
                    # 如果找不到对应图像，使用占位符
                    output_sequence.append(f"[IMAGE_{image_counter}]")
                
                image_counter += 1
                i = j + 1 if j < len(token_sequence) else j
                
            else:
                # 普通文本token，检查token_id是否有效
                if token_id is not None and isinstance(token_id, int):
                    text_buffer.append(token_id)
                else:
                    print(f"警告: 跳过无效的token_id: {token_id}")
                i += 1
        
        # 添加最后的文本片段
        if text_buffer:
            try:
                # 过滤掉None值
                valid_tokens = [t for t in text_buffer if t is not None and isinstance(t, int)]
                if valid_tokens:
                    text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    if text and text.strip():
                        output_sequence.append(text)
            except Exception as e:
                print(f"警告: 解码文本片段时出错: {e}")
                print(f"有问题的token序列: {text_buffer}")
        
        return output_sequence
    
    def create_unified_prompt(self, text_parts: List[str], image_placeholders: List[str] = None) -> str:
        """创建统一生成的提示，支持文本和图像占位符的交替排列
        
        Args:
            text_parts: 文本部分列表
            image_placeholders: 图像占位符列表，如["<img>", "<img>"]
            
        Returns:
            格式化的提示字符串
        """
        if image_placeholders is None:
            image_placeholders = []
            
        prompt_parts = []
        max_len = max(len(text_parts), len(image_placeholders))
        
        for i in range(max_len):
            if i < len(text_parts):
                prompt_parts.append(text_parts[i])
            if i < len(image_placeholders):
                # 使用特殊的图像开始token作为占位符
                img_start_token = self.tokenizer.decode([self.new_token_ids.get('start_of_image', 0)])
                if img_start_token:
                    prompt_parts.append(img_start_token)
                else:
                    prompt_parts.append(image_placeholders[i])
        
        return "".join(prompt_parts)
    
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
