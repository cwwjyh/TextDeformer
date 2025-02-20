#虚拟环境 kohya，可正常推理
import torch
from diffusers import FluxPipeline
import os
import yaml
import argparse
import pathlib

# 基础设置
pipe_id = "black-forest-labs/FLUX.1-dev"
lora_weight = "model_zoo/flux_lora"
lora_name = "mesh_nine_grid_672_clean_prompt_epo-step00005000.safetensors"

# 加载模型
pipe = FluxPipeline.from_pretrained(pipe_id, torch_dtype=torch.bfloat16)
print("Loading Lora weights")
pipe.load_lora_weights(lora_weight, weight_name=lora_name)
pipe.fuse_lora(lora_scale=1.0)
pipe.to("cuda")

# 图像设置
image_width = 672
image_height = 672



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', type=str, default='./example_config.yml')
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
    
    for key in vars(args):
        cfg[key] = vars(args)[key]
    output_path = pathlib.Path(cfg['output_path'])
    prompt = cfg['target_prompt']

    # 生成随机种子
    seed = torch.randint(0, 100000, (1,), device="cuda")[0].item()
    print(f"Using seed: {seed}")
    print("Generating image with prompt:", prompt)

    images = pipe(
        prompt,
        height=image_height,
        width=image_width,
        guidance_scale=7.5,
        num_inference_steps=20,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(seed),
        num_images_per_prompt=1,
    ).images

    # 保存结果，使用索引命名
    output_path = os.path.join(output_path, f"infer_flux.png")
    images[0].save(output_path)
    print(f"Image saved to: {output_path}")

    print("所有图片生成完成！")