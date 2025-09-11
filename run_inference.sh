#!/bin/bash
# 自回归交错生成模型推理启动脚本

# 设置默认参数
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/bagel/results/unified_training_20250910_201937/checkpoints/0000800"}
export OUTPUT_DIR=${OUTPUT_DIR:-"inference_outputs"}
export DEVICE=${DEVICE:-"cuda"}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 自回归交错生成模型推理工具${NC}"
echo "================================================="

# 检查检查点是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}❌ 检查点路径不存在: $CHECKPOINT_PATH${NC}"
    echo "请设置正确的检查点路径："
    echo "export CHECKPOINT_PATH=/path/to/your/checkpoint"
    exit 1
fi

echo -e "${GREEN}✅ 检查点路径: $CHECKPOINT_PATH${NC}"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}🔥 GPU信息:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1
else
    echo -e "${YELLOW}⚠️  未检测到GPU，将使用CPU（可能较慢）${NC}"
    export DEVICE="cpu"
fi

echo
echo "请选择推理模式："
echo "1. 快速测试 - 简单的文本/图像生成测试"
echo "2. 交互式生成 - 可以连续输入提示进行生成"
echo "3. 批量示例 - 运行预设的多个示例"
echo "4. 自定义命令 - 使用完整参数进行推理"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo -e "${BLUE}🎯 快速测试模式${NC}"
        read -p "请输入测试提示 (默认: 画一只可爱的小猫): " prompt
        prompt=${prompt:-"画一只可爱的小猫在花园里玩耍，阳光明媚"}
        
        echo -e "${YELLOW}正在执行快速测试...${NC}"
        python quick_inference.py \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --prompt "$prompt" \
            --output_dir "$OUTPUT_DIR/quick_test"
        ;;
    
    2)
        echo -e "${BLUE}💬 交互式生成模式${NC}"
        echo "启动交互式示例..."
        python inference_examples.py
        ;;
    
    3)
        echo -e "${BLUE}📚 批量示例模式${NC}"
        echo "运行所有预设示例..."
        
        mkdir -p "$OUTPUT_DIR"
        
        echo -e "${YELLOW}示例1: 文本转图像${NC}"
        python inference_unified_autoregressive.py \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --prompt "画一只橘猫在樱花树下玩耍，春天的午后，水彩画风格" \
            --mode autoregressive \
            --output_dir "$OUTPUT_DIR/example1_text_to_image" \
            --save_intermediate
        
        echo -e "${YELLOW}示例2: 多步骤生成${NC}"
        python inference_unified_autoregressive.py \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --prompt "从前有一个魔法森林，森林里住着智慧的老树精。画出这个神秘的森林。然后描述老树精的样子。最后画出老树精在森林中的场景。" \
            --mode step_by_step \
            --output_dir "$OUTPUT_DIR/example2_multi_step" \
            --save_intermediate
        
        # 如果有测试图像，运行图像编辑示例
        if [ -f "test_images/meme.jpg" ]; then
            echo -e "${YELLOW}示例3: 图像编辑${NC}"
            python inference_unified_autoregressive.py \
                --checkpoint_path "$CHECKPOINT_PATH" \
                --input_image "test_images/meme.jpg" \
                --prompt "添加彩虹背景。让图像变得更加梦幻。应用卡通动漫风格。" \
                --mode image_editing \
                --output_dir "$OUTPUT_DIR/example3_image_editing" \
                --save_intermediate
        fi
        ;;
    
    4)
        echo -e "${BLUE}⚙️  自定义命令模式${NC}"
        echo "请输入自定义参数："
        
        read -p "输入提示: " custom_prompt
        read -p "生成模式 (autoregressive/step_by_step/image_editing): " custom_mode
        read -p "最大长度 (默认500): " max_length
        read -p "图像宽度 (默认1024): " img_width
        read -p "图像高度 (默认1024): " img_height
        read -p "输入图像路径 (可选): " input_image
        
        max_length=${max_length:-500}
        img_width=${img_width:-1024}
        img_height=${img_height:-1024}
        custom_mode=${custom_mode:-autoregressive}
        
        cmd="python inference_unified_autoregressive.py \
            --checkpoint_path \"$CHECKPOINT_PATH\" \
            --prompt \"$custom_prompt\" \
            --mode $custom_mode \
            --max_length $max_length \
            --image_width $img_width \
            --image_height $img_height \
            --output_dir \"$OUTPUT_DIR/custom\" \
            --save_intermediate"
        
        if [ -n "$input_image" ] && [ -f "$input_image" ]; then
            cmd="$cmd --input_image \"$input_image\""
        fi
        
        echo -e "${YELLOW}执行命令: $cmd${NC}"
        eval $cmd
        ;;
    
    *)
        echo -e "${RED}无效选择，退出${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}🎉 推理完成！${NC}"
echo -e "结果保存在: ${BLUE}$OUTPUT_DIR${NC}"

# 显示输出目录内容
if [ -d "$OUTPUT_DIR" ]; then
    echo
    echo "生成的文件："
    find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.txt" -o -name "*.json" | head -10
    
    total_images=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
    echo -e "${GREEN}总共生成了 $total_images 张图像${NC}"
fi

echo
echo "如需查看详细使用说明，请参考: README_inference.md"
