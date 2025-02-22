#!/bin/bash

all_gpus=4

gpu_id=0  # 从GPU 0开始，逐步递增

# 默认的配置文件路径
DEFAULT_CONFIG="./config.yml"

# 如果传入参数，则使用传入的配置文件路径
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "配置文件 $CONFIG_FILE 不存在，请检查路径。"
  exit 1
fi

echo "正在使用配置文件: $CONFIG_FILE"

# 定义epochs和clip_weights数组
epochs=(2000 2500)
clip_weights=(1.0 2.0 3.0)

# 按批次处理
for ((i=0; i<${#epochs[@]}; i++)); do
  for ((j=0; j<${#clip_weights[@]}; j++)); do

    # 输出当前批次的信息
    echo "Processing batch with GPU $gpu_id; Epoch: ${epochs[$i]}; Clip Weight: ${clip_weights[$j]}"
        
    # 设置当前批次的 GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id

    # 执行 Python 脚本，并传递参数
    python main.py --config "$CONFIG_FILE" --epoch ${epochs[$i]} --clip_weight ${clip_weights[$j]} --delta_clip_weight ${clip_weights[$j]} --output_path "${CONFIG_FILE}_${epochs[$i]}_${clip_weights[$j]}" &
  
    # 每批次使用不同的 GPU，更新 GPU ID
    gpu_id=$(( (gpu_id + 1) % $all_gpus ))  # 假设最多有 8 个 GPU，如果有更多，可以调整为更大的数字
  done
done

# 等待所有后台任务完成
wait

echo "All batches are processed."
