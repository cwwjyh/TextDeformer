#!/bin/bash

# 设置一个计数器，统计每批次的配置文件
batch_size=8
gpu_id=0  # 从GPU 0开始，逐步递增

# 获取配置文件路径列表
config_files=(config/*.yml)

# 总的配置文件数
total_configs=${#config_files[@]}

# 按批次处理
for ((i=0; i<$total_configs; i+=$batch_size)); do
  # 获取当前批次的配置文件列表
  batch=("${config_files[@]:i:$batch_size}")
  
  # 输出当前批次的信息
  echo "Processing batch with GPU $gpu_id"
  
  # 遍历当前批次的所有配置文件
  for config_file in "${batch[@]}"; do
    if [ -f "$config_file" ]; then
      # 输出正在使用的配置文件名
      echo "Running with config file: $config_file on GPU $gpu_id"
      
      # 设置当前批次的 GPU
      export CUDA_VISIBLE_DEVICES=$gpu_id

      # 执行 Python 脚本，并传递 config 文件作为参数
      python main.py --config "$config_file" &
    else
      echo "Config file $config_file not found."
    fi
  done
  
  # 等待所有当前批次的进程完成
  wait
  
  # 每批次使用不同的 GPU，更新 GPU ID
  gpu_id=$(( (gpu_id + 1) % 8 ))  # 假设最多有 8 个 GPU，如果有更多，可以调整为更大的数字
done

echo "All batches are processed."
