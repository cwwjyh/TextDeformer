import torch
from diffusers import FluxPipeline
import os

# 基础设置
pipe_id = "black-forest-labs/FLUX.1-dev"
cache_dir = '/home/i-caiweiwei/.cache/huggingface/hub'
lora_weight = "/data/caiweiwei/kohya_ss/outputs/mesh_image_hippopotamus_no_pose/"
lora_name = "mesh_image_hippopotamus_no_pose-step00003000.safetensors"
output_dir = "/data/caiweiwei/kohya_ss/outputs/inference"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
pipe = FluxPipeline.from_pretrained(pipe_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
print("Loading Lora weights")
pipe.load_lora_weights(lora_weight, weight_name=lora_name)
pipe.fuse_lora(lora_scale=0.5)
pipe.to("cuda")

# 定义 SDS Loss
def sds_loss(pipe, generated_image, target_prompt, noise_level=0.05):
    """
    计算 SDS Loss。
    :param pipe: Fine-tune 后的 Flux 模型。
    :param generated_image: 生成模型的输出图像（张量形式）。
    :param target_prompt: 目标提示词。
    :param noise_level: 噪声水平。
    :return: SDS Loss 值。
    """
    # 添加噪声
    noise = torch.randn_like(generated_image) * noise_level
    noisy_image = generated_image + noise
    
    # 计算分数函数（梯度）
    noisy_image.requires_grad_(True)
    output = pipe(
        target_prompt,
        image=noisy_image,
        guidance_scale=7.5,
        num_inference_steps=20,
        max_sequence_length=512,
        return_dict=True,
    )
    score = output.images[0]  # 获取分数函数
    
    # 计算 SDS Loss
    loss = torch.mean((score - generated_image) ** 2)
    return loss

# 生成模型和优化器
class GeneratorModel(torch.nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 256 * 256 * 3),  # 假设输入是 512 维的向量
            torch.nn.Tanh(),  # 输出范围为 [-1, 1]
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 3, 256, 256)  # 输出图像形状为 (3, 256, 256)

generate_model = GeneratorModel().to("cuda")
optimizer = torch.optim.Adam(generate_model.parameters(), lr=1e-4)

# 训练循环
target_prompt = """
This image depicts a 2D mesh of a stylized unicorn captured from a side perspective. The mesh features a compact, rounded body with short, sturdy legs. The unicorn's head is distinct with a pronounced, curved snout and a single, prominent horn extending from its forehead, adding to its mythical and enchanting appearance. The small ears are positioned behind the horn, contributing to the whimsical style. The mesh exhibits a faceted, low-polygon style, giving the unicorn a geometric and simplified look. The absence of texture and color in the mesh suggests that it is a basic 3D model, potentially serving as a foundation for further design and development.
"""

for epoch in range(100):  # 假设训练 100 个 epoch
    # 生成输入数据（假设输入是随机向量）
    input = torch.randn(1, 512).to("cuda")
    
    # 生成图像
    generated_image = generate_model(input)
    
    # 计算 SDS Loss
    loss = sds_loss(pipe, generated_image, target_prompt)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# 保存生成的图像
output_path = os.path.join(output_dir, "generated_unicorn.png")
torchvision.utils.save_image(generated_image, output_path)
print(f"Image saved to: {output_path}")