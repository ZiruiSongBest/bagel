#!/bin/bash
# 统一生成模型训练快速启动脚本

echo "🚀 BAGEL统一生成模型训练快速启动"
echo "=================================="

# 检查Python环境
echo "检查Python环境..."
python --version

# 检查CUDA可用性
echo "检查CUDA可用性..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 创建输出目录
OUTPUT_DIR="./outputs/unified_training_$(date +%Y%m%d_%H%M%S)"
echo "创建输出目录: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

# 设置默认参数
MODEL_PATH="${MODEL_PATH:-models/BAGEL-7B-MoT}"
TRAIN_DATA="${TRAIN_DATA:-data/train.jsonl}"
VAL_DATA="${VAL_DATA:-data/val.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

echo "训练参数:"
echo "  模型路径: $MODEL_PATH"
echo "  训练数据: $TRAIN_DATA"
echo "  验证数据: $VAL_DATA"
echo "  批次大小: $BATCH_SIZE"
echo "  梯度累积: $GRAD_ACCUM"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  输出目录: $OUTPUT_DIR"

# 检查模型和数据文件
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    echo "请设置正确的MODEL_PATH环境变量或将模型放在默认路径"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据不存在: $TRAIN_DATA"
    echo "请设置正确的TRAIN_DATA环境变量或准备训练数据"
    echo "数据格式请参考: training/sample_data_format.json"
    exit 1
fi

echo "开始训练..."
echo "=================================="

# 执行训练
python train_unified_generation.py \
    --model_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA" \
    --val_data_path "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --fp16 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 200 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --text_loss_weight 1.0 \
    --image_loss_weight 1.0 \
    --save_total_limit 3

echo "=================================="
echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "查看训练日志和检查点以了解训练结果"
