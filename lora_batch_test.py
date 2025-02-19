#虚拟环境 kohya，可正常推理
import torch
from diffusers import FluxPipeline
import os

# 基础设置
pipe_id = "black-forest-labs/FLUX.1-dev"
# pipe_id = "model_zoo/flux"

lora_weight = "model_zoo/flux_lora"
lora_name = "mesh_nine_grid_672_clean_prompt_epo-step00005000.safetensors"

output_dir = "outputs/inference"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
pipe = FluxPipeline.from_pretrained(pipe_id, torch_dtype=torch.bfloat16)
print("Loading Lora weights")
pipe.load_lora_weights(lora_weight, weight_name=lora_name)
pipe.fuse_lora(lora_scale=1.0)
pipe.to("cuda")

# 图像设置
image_width = 1024
image_height = 1024

prompts = ["""The nine-grid image shows different views of the Bust of Venus. The
  Bust of Venus is characterized by a graceful, rounded form with an emphasis on smooth,
  flowing lines. The head features subtle contours that suggest elegant facial features,
  including a gentle curve representing the nose and a rounded structure indicating
  the chin. The hair is depicted with soft, abstract protrusions, capturing the essence
  of flowing locks in a simplified manner. The mesh exhibits a smooth, minimalistic
  style that gives the Bust of Venus an abstract and elegant appearance. The absence
  of texture and color in the mesh suggests that it is a basic 3D model, serving as
  a foundation for further artistic refinement or realistic detailing.""",
           ]

# 为每个prompt生成图像
for idx, prompt in enumerate(prompts):
    # 生成随机种子
    seed = torch.randint(0, 100000, (1,), device="cuda")[0].item()
    print(f"Processing prompt {idx+1}/{len(prompts)}")
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
    output_path = os.path.join(output_dir, f"generated_four_image_{idx+1}_scale1.0_5000saf_1024.png")
    images[0].save(output_path)
    print(f"Image saved to: {output_path}")

print("所有图片生成完成！")