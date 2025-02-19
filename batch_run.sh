#!/bin/bash

# 遍历当前目录下所有以 cow2 开头的 .yml 文件
for config_file in config/cow2*.yml; do
  if [ -f "$config_file" ]; then
    # 输出正在使用的配置文件名
    echo "Running with config file: $config_file"

    # 执行 Python 脚本，并传递 config 文件作为参数
    python render_source_mesh.py --config "$config_file"
    python gpt-4o.py --config "$config_file"

    # 可选：如果需要，添加其他逻辑，如日志记录等
  else
    echo "No files matching cow2*.yml found."
  fi
done
