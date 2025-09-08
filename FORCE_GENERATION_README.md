# BAGEL 强制图像生成功能

## 问题背景

原始的统一生成功能期望模型能够自然地生成特殊的图像token（如`<|vision_start|>`），但由于模型没有经过专门训练，在实际使用中模型不会自发生成这些token。这导致图像生成功能无法被触发。

## 解决方案

我们实现了几种方法来解决这个问题：

### 1. 改进的系统提示词

- **VLM_THINK_SYSTEM_PROMPT**: 用于理解任务
- **GEN_THINK_SYSTEM_PROMPT**: 用于图像生成，包含了图像token的示例
- **UNIFIED_GEN_SYSTEM_PROMPT**: 新增的统一生成系统提示词，明确指导模型使用特殊token

### 2. 强制图像生成方法

#### `test_force_image_generation()`
直接跳过token生成过程，强制触发图像生成。适用于测试和确保图像生成功能正常工作。

```python
result = inferencer.test_force_image_generation(
    input_text="A beautiful sunset",
    image_shapes=(1024, 1024),
    cfg_text_scale=4.0,
    cfg_img_scale=1.5,
    num_timesteps=50,
)
```

#### `unified_generate()` 改进版
添加了 `force_image_generation` 参数，可以自动在输入文本后添加图像token。

```python
result = inferencer.unified_generate(
    input_text="Generate a landscape",
    force_image_generation=True,  # 强制添加图像token
    use_unified_system_prompt=True,  # 使用改进的系统提示词
)
```

### 3. 手动Token注入

可以在提示词中手动添加特殊token：

```python
prompt = "Create a beautiful image: <|vision_start|> <|vision_end|>"
result = inferencer.unified_generate(input_text=prompt)
```

## 使用方法

### 测试脚本

运行测试脚本来验证功能：

```bash
# 基础测试
python test_force_generation.py

# 使用不同模型路径
python test_force_generation.py --model_path /path/to/your/model

# 使用量化模式
python test_force_generation.py --mode 2  # NF4量化

# 自定义测试提示词
python test_force_generation.py --prompt "A dragon flying over a castle"
```

### 图像编辑Demo

```bash
# 测试强制生成功能
python image_editing_demo.py --test_force_generation --prompt "A magical forest"

# 测试unified_generate方法
python image_editing_demo.py --test_unified --force_image_gen --prompt "Edit this image"

# 标准图像编辑（使用interleave_inference）
python image_editing_demo.py --input_image cat.jpg --prompt "Add a hat to the cat"
```

## 参数说明

### `test_force_image_generation()` 参数
- `input_text`: 输入文本提示
- `image_shapes`: 图像尺寸，默认(1024, 1024)
- `cfg_text_scale`: 文本CFG缩放因子，默认4.0
- `cfg_img_scale`: 图像CFG缩放因子，默认1.5
- `num_timesteps`: 去噪步数，默认50

### `unified_generate()` 新增参数
- `force_image_generation`: 是否强制生成图像，默认False
- `use_unified_system_prompt`: 是否使用改进的系统提示词，默认True

## 特殊Token映射

确保以下token正确配置：

```python
new_token_ids = {
    'start_of_image': tokenizer.convert_tokens_to_ids('<|vision_start|>'),
    'end_of_image': tokenizer.convert_tokens_to_ids('<|vision_end|>'),
    'bos_token_id': tokenizer.convert_tokens_to_ids('<|im_start|>'),
    'eos_token_id': tokenizer.convert_tokens_to_ids('<|im_end|>'),
}
```

## 调试技巧

1. **检查特殊token**: 使用测试脚本验证所有特殊token都正确映射
2. **降低生成参数**: 测试时使用较小的图像尺寸和较少的时间步数以加快速度
3. **监控显存**: 图像生成需要大量显存，确保有足够的GPU内存
4. **查看生成日志**: 启用详细日志来跟踪生成过程

## 局限性

1. **训练依赖**: 最理想的解决方案是对模型进行训练，让它学会自然生成图像token
2. **提示词依赖**: 当前方案依赖于改进的提示词，但模型可能不会总是遵循指令
3. **性能开销**: 强制方法绕过了正常的token生成流程，可能不如端到端训练的解决方案高效

## 下一步改进

1. **Few-shot学习**: 在提示词中提供更多的示例来教导模型使用特殊token
2. **LoRA微调**: 使用少量数据对模型进行轻量级微调
3. **Token概率调整**: 在生成过程中调整特殊token的概率分布
4. **上下文学习**: 通过精心设计的上下文来引导模型行为






