#!/bin/bash


# for config_file in config/*.yml; do
#   if [ -f "$config_file" ]; then
#     # 输出正在使用的配置文件名
#     echo "Running with config file: $config_file"

#     # 执行 Python 脚本，并传递 config 文件作为参数
#     python infer_flux_lora.py --config "$config_file"

#     # 可选：如果需要，添加其他逻辑，如日志记录等
#   else
#     echo "No files matching *.yml found."
#   fi
# done

# for config_file in config/*.yml; do
#   if [ -f "$config_file" ]; then
#     # 输出正在使用的配置文件名
#     echo "Running with config file: $config_file"

#     # 执行 Python 脚本，并传递 config 文件作为参数
#     python gpt-4o.py --config "$config_file"

#     # 可选：如果需要，添加其他逻辑，如日志记录等
#   else
#     echo "No files matching *.yml found."
#   fi
# done

batch_size = [1,2,3,4]

for config_file in config/*.yml; do
  if [ -f "$config_file" ]; then
    # 输出正在使用的配置文件名
    echo "Running with config file: $config_file"

    # 执行 Python 脚本，并传递 config 文件作为参数
    python main.py --config "$config_file"

    # 可选：如果需要，添加其他逻辑，如日志记录等
  else
    echo "No files matching *.yml found."
  fi
done