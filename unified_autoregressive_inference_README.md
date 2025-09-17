# 统一自回归推理说明

## 核心理念

BAGEL的统一自回归推理完全对应训练时的逻辑，实现真正的交错生成：

```
文本解码 → 预测<vision_start> → 逐patch图像生成 → 预测<vision_end> → 继续文本解码
```

## 与训练的对应关系

### 1. 训练时的逻辑（bagel.py中的forward_autoregressive_training）

```python
# 训练时的统一序列：
# "分析这张图片 <vision_start> [patch_0] [patch_1] ... [patch_n] <vision_end> 结果是..."

# 逐token处理：
for token in target_sequence:
    if token == text_token:
        # 计算文本CE loss
    elif token == '<vision_start>':
        # 计算特殊token CE loss（加权）
    elif token == image_patch:
        # 计算Flow Matching loss
    elif token == '<vision_end>':
        # 计算特殊token CE loss
```

### 2. 推理时的逻辑（unified_image_editing_inference.py）

```python
# 推理时的逐token生成：
while not done:
    # 1. 预测下一个token类型
    if not in_image_generation:
        # 预测文本token或<vision_start>
        next_token = model.predict_next()
        if next_token == '<vision_start>':
            in_image_generation = True
    else:
        # 在图像生成中
        if patches_generated < max_patches:
            # 生成下一个patch（Flow Matching）
            patch = generate_patch_with_flow_matching()
        else:
            # 预测<vision_end>
            next_token = '<vision_end>'
            in_image_generation = False
```

## 关键创新点

### 1. 模型自主决策
- 模型自己决定何时开始图像生成（预测`<vision_start>`）
- 模型自己决定何时结束图像生成（预测`<vision_end>`）
- 不需要外部控制或强制插入特殊token

### 2. 逐Patch生成
- 每个图像patch都是独立生成的
- 使用Flow Matching进行去噪
- 保持与训练时相同的序列建模方式

### 3. 统一的序列建模
- 文本、特殊token、图像patch在同一个序列中
- 使用相同的自回归机制
- 训练和推理完全一致

## 使用示例

```python
from unified_image_editing_inference import UnifiedImageEditingInference

# 初始化
engine = UnifiedImageEditingInference(model_path="path/to/model")

# 图像编辑（统一自回归模式）
results = engine.unified_edit_image(
    image_path="input.jpg",
    edit_prompt="将天空变成夕阳",
    use_autoregressive=True,  # 使用统一自回归
    max_length=500,
    do_sample=True,
    temperature=0.8
)

# 多模态生成
results = engine.autoregressive_multi_modal_generation(
    prompt="基于这张图片生成艺术作品",
    input_image="reference.jpg",
    force_image_generation=True
)
```

## 技术细节

### Flow Matching在推理中的实现

```python
def generate_patch_with_flow_matching(self, ...):
    # 1. 从纯噪声开始
    x_t = torch.randn(...)
    
    # 2. 逐步去噪
    for t in timesteps:
        # 预测velocity
        v_pred = model.predict_velocity(x_t, t)
        # 更新
        x_t = x_t - v_pred * dt
    
    # 3. 得到clean patch
    return x_t
```

### 特殊Token的权重

训练时对`<vision_start>`等特殊token给予更高权重，确保模型学会正确的时序控制：

```python
if token_type == 'special' and token_id == start_of_image:
    loss = loss * 2.0  # 更高权重
```

## 优势

1. **训练推理一致性**：完全对应训练逻辑，减少分布偏移
2. **灵活性**：模型自主决定生成内容和时机
3. **统一建模**：文本和图像在同一框架下处理
4. **渐进式生成**：逐token/逐patch生成，可控性强

## 注意事项

1. 生成速度相对较慢（逐patch生成）
2. 需要足够的显存（保存完整序列）
3. 温度参数对生成质量影响较大
4. Flow Matching步数影响图像质量和速度的权衡
