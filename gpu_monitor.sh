#!/bin/bash

# GPU监控和自动训练启动脚本
# 监控4张显卡，当全部空闲时自动启动训练

# 配置参数
CHECK_INTERVAL=30  # 检查间隔(秒)
GPU_COUNT=4        # 监控的GPU数量
GPU_MEMORY_THRESHOLD=500  # GPU内存使用阈值(MB)，低于此值认为空闲

# 日志文件
LOG_FILE="./gpu_monitor.log"
TRAINING_SCRIPT="./unitrain_modelparallel.sh"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 记录日志函数
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - ${message}" | tee -a "$LOG_FILE"
}

# 检查GPU状态函数
check_gpu_status() {
    local all_gpus_free=true
    
    echo -e "${BLUE}=== GPU状态检查 ===${NC}"
    
    for ((i=0; i<$GPU_COUNT; i++)); do
        # 获取GPU内存使用情况
        local gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        local gpu_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $i)
        
        echo -e "GPU $i: 内存使用 ${gpu_memory}MB, 利用率 ${gpu_utilization}%"
        
        # 检查是否有进程在使用GPU
        local gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i | wc -l)
        
        if [ "$gpu_memory" -gt "$GPU_MEMORY_THRESHOLD" ] || [ "$gpu_processes" -gt 0 ]; then
            all_gpus_free=false
            echo -e "${RED}GPU $i 正在被使用${NC}"
        else
            echo -e "${GREEN}GPU $i 空闲${NC}"
        fi
    done
    
    if [ "$all_gpus_free" = true ]; then
        return 0  # 所有GPU都空闲
    else
        return 1  # 有GPU在使用
    fi
}

# 启动训练函数
start_training() {
    log_message "${GREEN}所有GPU都空闲，开始启动训练...${NC}"
    
    # 检查训练脚本是否存在
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        log_message "${RED}错误: 训练脚本 $TRAINING_SCRIPT 不存在${NC}"
        return 1
    fi
    
    # 使脚本可执行
    chmod +x "$TRAINING_SCRIPT"
    
    # 启动训练
    log_message "${GREEN}执行训练脚本: $TRAINING_SCRIPT${NC}"
    bash "$TRAINING_SCRIPT"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_message "${GREEN}训练完成${NC}"
    else
        log_message "${RED}训练出现错误，退出码: $exit_code${NC}"
    fi
    
    return $exit_code
}

# 显示帮助信息
show_help() {
    echo "GPU监控和自动训练启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示帮助信息"
    echo "  -i, --interval SECONDS  设置检查间隔(默认: 30秒)"
    echo "  -t, --threshold MB      设置GPU内存阈值(默认: 500MB)"
    echo "  -c, --count NUMBER      设置监控的GPU数量(默认: 4)"
    echo "  -s, --script PATH       设置训练脚本路径(默认: ./unitrain_modelparallel.sh)"
    echo ""
    echo "示例:"
    echo "  $0                      使用默认设置开始监控"
    echo "  $0 -i 60 -t 1000        每60秒检查一次，内存阈值1000MB"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        -t|--threshold)
            GPU_MEMORY_THRESHOLD="$2"
            shift 2
            ;;
        -c|--count)
            GPU_COUNT="$2"
            shift 2
            ;;
        -s|--script)
            TRAINING_SCRIPT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主监控循环
main() {
    log_message "${BLUE}开始GPU监控...${NC}"
    log_message "配置: 检查间隔=${CHECK_INTERVAL}秒, GPU数量=${GPU_COUNT}, 内存阈值=${GPU_MEMORY_THRESHOLD}MB"
    log_message "训练脚本: $TRAINING_SCRIPT"
    
    # 检查nvidia-smi是否可用
    if ! command -v nvidia-smi &> /dev/null; then
        log_message "${RED}错误: nvidia-smi 命令不可用${NC}"
        exit 1
    fi
    
    # 检查GPU数量
    local available_gpus=$(nvidia-smi -L | wc -l)
    if [ "$available_gpus" -lt "$GPU_COUNT" ]; then
        log_message "${YELLOW}警告: 可用GPU数量($available_gpus) 小于配置数量($GPU_COUNT)${NC}"
        GPU_COUNT=$available_gpus
    fi
    
    while true; do
        local current_time=$(date '+%H:%M:%S')
        echo -e "\n${BLUE}[$current_time] 检查GPU状态...${NC}"
        
        if check_gpu_status; then
            # 所有GPU都空闲，启动训练
            start_training
            
            # 训练完成后退出监控（如果你想继续监控，注释掉下面这行）
            log_message "${GREEN}训练任务完成，退出监控${NC}"
            break
        else
            # 有GPU在使用，继续等待
            echo -e "${YELLOW}有GPU正在使用中，${CHECK_INTERVAL}秒后再次检查...${NC}"
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# 捕获中断信号
trap 'log_message "${YELLOW}监控被中断${NC}"; exit 0' SIGINT SIGTERM

# 启动主程序
main "$@"
