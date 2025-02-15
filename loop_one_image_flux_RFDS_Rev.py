#虚拟环境kohya_new1
#loop_one_image_flux.py
"""
1. vae_scale_factor 是否需要调节？
2. 噪声调度器（noise_scheduler）是否正确初始化? 已修正，与fintune flux用的noise_scheduler
3. cfg.flux_weight设置是否合理
4. pipe.fuse_lora(lora_scale=0.5)中的lora_scale是非常关键的参数，可以调节
5. 仿照RFDS_Rev_sd3.py的noise预测进行修改这里的flux loss
6. 文件命名：loop_one_image_flux_RFDS_Rev.py
"""
import clip
import kornia
import os
import pathlib
import pymeshlab
import shutil
import torch
import torchvision
import logging
import yaml

import numpy as np
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt

from easydict import EasyDict

from NeuralJacobianFields import SourceMesh

from nvdiffmodeling.src     import obj
from nvdiffmodeling.src     import util
from nvdiffmodeling.src     import mesh
from nvdiffmodeling.src     import render
from nvdiffmodeling.src     import texture
from nvdiffmodeling.src     import regularizer

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utilities.video import Video
from utilities.helpers import cosine_avg, create_scene, get_vp_map
from utilities.camera import CameraBatch, get_camera_params
from utilities.clip_spatial import CLIPVisualEncoder
from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3
# from IPython import embed
# from src.guidance.stable_diffusion import StableDiffusionGuidance

#cww add flux
from diffusers import DDPMScheduler  # 确保正确导入 DDPMScheduler
#load flux model sds loss
from diffusers import FluxPipeline
import library.train_util as train_util
from library import model_util
import importlib
from accelerate import Accelerator
from transformers import CLIPTokenizer
import sys
sys.path.append('/data/caiweiwei/TextDeformer-main/')
import library
print(library.__file__)
# from library import strategy_flux
from library import (
    flux_models,
    flux_train_utils,
    flux_utils,
    # strategy_base,
    strategy_flux,
    sd3_train_utils
)
from library.custom_train_functions import (
    # apply_snr_weight,
    # get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    # scale_v_prediction_loss_like_noise_prediction,
    # add_v_prediction_like_loss,
    # apply_debiased_estimation,
    # apply_masked_loss,
)
# 基础设置
pipe_id = "black-forest-labs/FLUX.1-dev"
cache_dir = '/home/i-caiweiwei/.cache/huggingface/hub'
lora_weight = "/data/caiweiwei/kohya_ss/outputs/mesh_image_hippopotamus_no_pose/"
lora_name = "mesh_image_hippopotamus_no_pose-step00003000.safetensors"
# output_dir = "/data/caiweiwei/kohya_ss/outputs/inference"
# os.makedirs(output_dir, exist_ok=True)

# # 设置环境变量以避免内存碎片
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 加载模型
pipe = FluxPipeline.from_pretrained(pipe_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
print("Loading Lora weights")
pipe.load_lora_weights(lora_weight, weight_name=lora_name)
pipe.fuse_lora(lora_scale=1.0) #这个可以调节
pipe.to("cuda")
print("loaded lora weights end")


# 定义获取噪声调度器的函数
#调用flux_train_network.py,而不是train_network.py
def get_noise_scheduler(device: torch.device):
    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    # breakpoint()
    return noise_scheduler


def call_unet(unet, noisy_latents, timesteps, text_conds, **kwargs):
    noise_pred = unet(noisy_latents, timesteps, text_conds).sample
    return noise_pred

def get_models_for_text_encoding(args, accelerator, text_encoders):
    if args.cache_text_encoder_outputs:
        if self.train_clip_l and not self.train_t5xxl:
            return text_encoders[0:1]  # only CLIP-L is needed for encoding because T5XXL is cached
        else:
            return None  # no text encoders are needed for encoding because both are cached
    else:
        return text_encoders  # both CLIP-L and T5XXL are needed for encoding
def get_noise_pred_and_target_Rev(
        args,
        accelerator,
        noise_scheduler,
        latents,
        # batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        # network,
        weight_dtype,
        train_unet,
    ):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents) #torch.Size([1, 16, 28, 28])
        bsz = latents.shape[0] #bsz=1
        # breakpoint()
        # get noisy model input and timesteps（pdb查看跳到哪里去了）此处的输出与RFDS_sd3.py的输出是一样的
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        # pack latents and get img_ids
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # get guidance
        # ensure guidance_scale in args is float
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

        # Predict the noise residual
        l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
        if not args.apply_t5_attn_mask: #这里打开了
            t5_attn_mask = None

        def call_dit(img, img_ids, t5_out, txt_ids, l_pooled, timesteps, guidance_vec, t5_attn_mask):
            # if not args.split_mode:
            # normal forward
            # breakpoint() #想看unet的结构是什么？
            with accelerator.autocast():
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                model_pred = unet(  #jump flux_model.py Flux,class DoubleStreamBlock的forward()
                    img=img,  #shape = torch.Size([1, 196, 64])
                    img_ids=img_ids,  #torch.Size([1, 196, 3])
                    txt=t5_out, #shape=torch.Size([1, 77, 4096])
                    txt_ids=txt_ids, #torch.Size([1, 77, 3])
                    y=l_pooled, #torch.Size([1, 768])
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec, #torch.Size([1])
                    txt_attention_mask=t5_attn_mask,#torch.Size([1, 154])  这里必须t5_attn_mask=none,否则会报q，k与v维度不匹配问题
                )

            return model_pred

        model_pred = call_dit(
            img=packed_noisy_model_input,
            img_ids=img_ids,
            t5_out=t5_out.cuda(),
            txt_ids=txt_ids.cuda(),
            l_pooled=l_pooled.cuda(),
            timesteps=timesteps,
            guidance_vec=guidance_vec,
            t5_attn_mask=t5_attn_mask,
        )

        # unpack latents
        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        print("use iRFDS")
        stepsize = 1 - sigmas
        noise = noise + stepsize * (model_pred + latents - noise) #the optimized noise after 1 iRFDS step（论文中running iRFDS for just one step yields very satisfactory results）,接着RFDS的输入是iRFDS优化后的noise，而不是随机初始化的nosie

        ###############################################################
        # RFDS
        # get noisy model input and timesteps（pdb查看跳到哪里去了）此处的输出与RFDS_sd3.py的输出是一样的
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        # pack latents and get img_ids
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        model_pred = call_dit(
            img=packed_noisy_model_input,
            img_ids=img_ids,
            t5_out=t5_out.cuda(),
            txt_ids=txt_ids.cuda(),
            l_pooled=l_pooled.cuda(),
            timesteps=timesteps,
            guidance_vec=guidance_vec,
            t5_attn_mask=t5_attn_mask,
        )

        # unpack latents
        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        u = torch.normal(mean=0, std=1, size=(bsz,), device=accelerator.device)
        weighting = torch.nn.functional.sigmoid(u)
        # breakpoint()

        # flow matching loss: this is different from SD3
        # target = noise - latents

        return model_pred, noise, sigmas, weighting  #output：donkey2horse_flux_one_image_5000_fix_hyp_bs1_test2 此处没有用CFG
        # return noise_pred, noise, sigmas, weighting  #使用CFG，output：donkey2horse_flux_one_image_5000_fix_hyp_bs1_test2_modify但是效果没有不使用CFG效果好，因为flux本身没有使用CFG


#摘自flux_train_network.py
def load_target_model(args, weight_dtype):
        # currently offload to cpu for some models

        # if the file is fp8 and we are using fp8_base, we can load it as is (fp8)
        loading_dtype = None if args.fp8_base else weight_dtype

        # if we load to cpu, flux.to(fp8) takes a long time, so we should load to gpu in future
        _ , model = flux_utils.load_flow_model(
            args.pretrained_model_name_or_path, loading_dtype, "cpu", disable_mmap=False
        )

        # self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        # if self.is_swapping_blocks:
        #     # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
        #     logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        #     model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        print("clip_l path: ", args.clip_l)
        clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        clip_l.eval()

        # if the file is fp8 and we are using fp8_base (not unet), we can load it as is (fp8)
        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None  # as is
        else:
            loading_dtype = weight_dtype

        # loading t5xxl to cpu takes a long time, so we should load to gpu in future
        t5xxl = flux_utils.load_t5xxl(args.t5xxl, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        t5xxl.eval()
        # if args.fp8_base and not args.fp8_base_unet:
        #     # check dtype of model
        #     if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
        #         raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
        #     elif t5xxl.dtype == torch.float8_e4m3fn:
        #         logger.info("Loaded fp8 T5XXL model")

        # ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)

        return [clip_l, t5xxl], model
def prepare_unet_with_accelerator(accelerator: Accelerator, unet: torch.nn.Module) -> torch.nn.Module:
    return accelerator.prepare(unet)

def shift_scale_latents(latents, vae_scale_factor):
    return latents * vae_scale_factor #self.vae_scale_factor = 0.18215

#摘自flux_train_network.py
def get_tokenize_strategy(args):
    _, is_schnell, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)

    if args.t5xxl_max_token_length is None:
        if is_schnell:
            t5xxl_max_token_length = 256
        else:
            t5xxl_max_token_length = 512
    else:
        t5xxl_max_token_length = args.t5xxl_max_token_length

    # logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
    return strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length, None)

def encode_prompt_with_pipe(cfg, pipe, prompt, accelerator, text_encoders):
    """
    使用 FluxPipeline 的 text_encoder 和 tokenizer 对 prompt 进行编码。
    :param pipe: 加载权重后的 FluxPipeline 对象。
    :param prompt: 输入的文本提示词。
    :return: 文本嵌入 (text_embedding)。
    """
    # 使用 tokenizer 对文本进行编码
    # breakpoint()  #为什么我这里得到的input_ids与kohyass的code（input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]）的少了一个 t5_tokens的key，vale。
    input_ids_1 = pipe.tokenizer( #type(input_ids)=transformers.tokenization_utils_base.BatchEncoding, len(input_ids)=2
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(pipe.device)
    input_ids_2 = pipe.tokenizer_2( #type(input_ids)=transformers.tokenization_utils_base.BatchEncoding, len(input_ids)=2
        prompt,
        padding="max_length",
        max_length=cfg.t5xxl_max_token_length,
        truncation=True,
        return_tensors="pt"
    ).to(pipe.device)
    # breakpoint()

    # 创建新的字典
    input_ids = [input_ids_1['input_ids'],input_ids_2['input_ids'], input_ids_2['attention_mask']]

    tokenize_strategy = get_tokenize_strategy(cfg) 
    # apply_t5_attn_mask = None
    # breakpoint()
    encoded_text_encoder_conds = strategy_flux.FluxTextEncodingStrategy.encode_tokens( #runing
                                tokenize_strategy,
                                get_models_for_text_encoding(cfg, accelerator, text_encoders),
                                input_ids,
                            )
    text_encoder_conds = encoded_text_encoder_conds
    return text_encoder_conds

# class SpecifyGradient(torch.autograd.Function):
#     # @staticmethod
#     # @custom_fwd
#     def forward(ctx, input_tensor, gt_grad):
#         ctx.save_for_backward(gt_grad) #ctx.requireds_grad=input_tensor=True; gt_grad=False
#         # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
#         return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

#     # @staticmethod
#     # @custom_bwd
#     def backward(ctx, grad_scale):
#         gt_grad, = ctx.saved_tensors
#         gt_grad = gt_grad * grad_scale #gt_grad.shape=torch.Size([1, 16, 28, 28])
#         # breakpoint()
#         return gt_grad, None

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.
    """
    if weighting_scheme == "sigma_sort":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def loop(cfg):
    output_path = pathlib.Path(cfg['output_path'])
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / 'config.yml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    cfg = EasyDict(cfg)
    
    print(f'Output directory {cfg.output_path} created')

    device = torch.device(f'cuda:{cfg.gpu}')
    torch.cuda.set_device(device)

    print('Loading CLIP Models')
    model, _ = clip.load(cfg.clip_model, device=device)
    fe = CLIPVisualEncoder(cfg.consistency_clip_model, cfg.consistency_vit_stride, device) #初始化CLIP模型，并将其移动到指定的设备

    clip_mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device) #定义CLIP图像的均值，用于后续的图像归一化
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device) #定义CLIP 图像的标准差
    # output video
    video = Video(cfg.output_path)

    # GL Context
    # glctx = dr.RasterizeGLContext()
    glctx = dr.RasterizeCudaContext() #创建一个 CUDA 上下文，用于图形渲染。

    print(f'Target text prompt is {cfg.text_prompt}')
    print(f'Base text prompt is {cfg.base_text_prompt}')
    with torch.no_grad():
        text_embeds = clip.tokenize(cfg.text_prompt).to(device)
        base_text_embeds = clip.tokenize(cfg.base_text_prompt).to(device)
        text_embeds = model.encode_text(text_embeds).detach()
        target_text_embeds = text_embeds.clone() / text_embeds.norm(dim=1, keepdim=True)

        delta_text_embeds = text_embeds - model.encode_text(base_text_embeds)
        delta_text_embeds = delta_text_embeds / delta_text_embeds.norm(dim=1, keepdim=True)

    os.makedirs(output_path / 'tmp', exist_ok=True)
    ms = pymeshlab.MeshSet() #初始化一个新的 MeshSet 对象，用于处理网格。
    ms.load_new_mesh(cfg.mesh) #加载配置中指定的网格文件。

    if cfg.retriangulate:
        print('Retriangulating shape')
        ms.meshing_isotropic_explicit_remeshing()
    
    if not ms.current_mesh().has_wedge_tex_coord(): #检查当前网格是否具有楔形纹理坐标。
        # some arbitrarily high number
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000) #如果没哟没有，则计算纹理坐标参数
    
    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'))

    load_mesh = obj.load_obj(str(output_path / 'tmp' / 'mesh.obj'))
    # breakpoint()
    load_mesh = mesh.unit_size(load_mesh) #将加载的网格调整为单位大小
    
    #load_mesh.v_pose为source mesh的顶点， load_mesh.t_pos_idx为source mesh的面片
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'), save_vertex_color=False) #再次保存当前网格，不保存顶点颜色。

    # Determine guidance prompt
    # prompt = f"a photo of {cfg.text_prompt}" #before writting
    prompt = f"a photo of sks {cfg.text_prompt}"
    print(f"Guidance Prompt: {str(prompt)}")
    print("target_prompt:", cfg.target_prompt)
    # embed()

    # TODO: Need these for rendering even if we don't optimize textures
    texture_map = texture.create_trainable(np.random.uniform(size=[512]*2 + [3], low=0.0, high=1.0), [512]*2, True)
    normal_map = texture.create_trainable(np.array([0, 0, 1]), [512]*2, True)
    specular_map = texture.create_trainable(np.array([0, 0, 0]), [512]*2, True)

    # breakpoint()
    load_mesh = mesh.Mesh(
        material={
            'bsdf': cfg.bsdf,
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=load_mesh # Get UVs from original loaded mesh
    )
    # breakpoint()
    #将输入的mesh初始化雅可比对象
    jacobian_source = SourceMesh.SourceMesh(0, str(output_path / 'tmp' / 'mesh.obj'), {}, 1, ttype=torch.float)
    if len(list((output_path / 'tmp').glob('*.npz'))) > 0:
        logging.warn(f'Using existing Jacobian .npz files in {str(output_path)}/tmp/ ! Please check if this is intentional.') 
    jacobian_source.load()
    jacobian_source.to(device)

    with torch.no_grad():
        # breakpoint()
        #从加载的网格顶点计算雅可比矩阵
        gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0)) #load_mesh.v_pos.shape=torch.Size([1584, 3])  torch.Size([1, 398, 3])的时候报错
    gt_jacobians.requires_grad_(True) #设置雅可比矩阵以计算梯度。

    optimizer = torch.optim.Adam([gt_jacobians], lr=cfg.lr) #初始化 Adam 优化器，优化雅可比矩阵。
    cams_data = CameraBatch( #初始化相机 有batch_size个相机参数
        cfg.train_res,
        [cfg.dist_min, cfg.dist_max],
        [cfg.azim_min, cfg.azim_max],
        [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
        [cfg.fov_min, cfg.fov_max],
        cfg.aug_loc,
        cfg.aug_light,
        cfg.aug_bkg,
        cfg.batch_size,
        rand_solid=True
    )
    cams = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True) #创建相机数据加载器。
    best_losses = {'CLIP': np.inf, 'total': np.inf}

    for out_type in ['final', 'best_clip', 'best_total']:
        os.makedirs(output_path / f'mesh_{out_type}', exist_ok=True)
    os.makedirs(output_path / 'images', exist_ok=True)
    logger = SummaryWriter(str(output_path / 'logs'))

    rot_ang = 0.0
    t_loop = tqdm(range(cfg.epochs), leave=False)

    if cfg.resize_method == 'cubic':
        resize_method = cubic
    elif cfg.resize_method == 'linear':
        resize_method = linear
    elif cfg.resize_method == 'lanczos2':
        resize_method = lanczos2
    elif cfg.resize_method == 'lanczos3':
        resize_method = lanczos3

     #cww add
    # 在循环开始之前，加载参考图像
    # reference_image_path = cfg.reference_image_path  # 从配置中获取参考图像路径
    # reference_image = Image.open(reference_image_path).convert('RGB')  # 加载图像并转换为RGB格式
    # reference_image = torchvision.transforms.ToTensor()(reference_image).unsqueeze(0).to(device)  # 转换为张量并添加批次维度
    # reference_image = resize(reference_image, out_shape=(224, 224), interp_method=resize_method)  # 确保参考图像的大小与训练渲染图像一致
    # reference_image = reference_image.repeat(10, 1, 1, 1)  # 在第0个维度上重复25次
    # print("reference_image.shap:", reference_image.shape) #(1,3,224,224)

    #cww add flux
    accelerator = train_util.prepare_accelerator(cfg)
    weight_dtype = torch.bfloat16 #weight_dtype = torch.bfloat16
    text_encoders, unet= load_target_model(cfg, weight_dtype) #加载模型，获得unet
    # weight_dtype, save_dtype = torch.bfloat16 #weight_dtype = torch.bfloat16
    vae_dtype = weight_dtype
    unet_weight_dtype = te_weight_dtype = weight_dtype
    unet.requires_grad_(False)
    unet.to(accelerator.device,dtype=unet_weight_dtype) # move to device because unet is not prepared by accelerator #此处报oom
    unet.eval()
    
    vae_scale_factor = 0.18215

    #  在你的代码中调用 get_noise_scheduler 函数
    # 定义噪声调度器
    print("这里操作时间长吗？")
    noise_scheduler = get_noise_scheduler(accelerator.device)
    # 使用加载了 LoRA 权重的模型进行文本编码
    # breakpoint()
    print("noise_schieduler运行时间不长")
    with torch.set_grad_enabled(False), accelerator.autocast():
        text_encoder_conds = encode_prompt_with_pipe(cfg, pipe, cfg.target_prompt, accelerator,text_encoders) #这个要放在for循环之外
    # breakpoint()
    print("text_encoder_conds运行的时间比较长？")
    # prepare network
    train_unet=False
    l2_loss = 0.0
    sds_loss = 0.0
    for it in t_loop:
        # cams = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True) #创建相机数据加载器。
         # 获取 cams 的大小
        print(f"cams size: {len(cams.dataset)}")  # 输出数据集的大小
        # updated vertices from jacobians
        # breakpoint()
        n_vert = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze() #从雅可比矩阵中提取更新后的顶点位置，并去掉多余的维度。

        # TODO: More texture code required to make it work ...
        ready_texture = texture.Texture2D(  #对材质的漫反射纹理（kd）应用高斯模糊，生成一个新的2D纹理。
            kornia.filters.gaussian_blur2d(
                load_mesh.material['kd'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )
        #创建一个全为0.5的纹理，作为没有纹理的材质。
        kd_notex = texture.Texture2D(torch.full_like(ready_texture.data, 0.5))

        ready_specular = texture.Texture2D( #对材质的镜面反射纹理（ks）应用高斯模糊，生成一个新的2D纹理。
            kornia.filters.gaussian_blur2d(
                load_mesh.material['ks'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )

        ready_normal = texture.Texture2D( #对法线纹理应用高斯模糊，生成一个新的2D纹理。
            kornia.filters.gaussian_blur2d(
                load_mesh.material['normal'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )
            
        # Final mesh
        # breakpoint()
        #创建一个新的网格对象m，使用更新后的顶点、面片索引和材质属性。
        m = mesh.Mesh( 
            n_vert,
            load_mesh.t_pos_idx,
            material={
                'bsdf': cfg.bsdf,
                'kd': kd_notex,
                'ks': ready_specular,
                'normal': ready_normal,
            },
            base=load_mesh # gets uvs etc from here
        )
        # breakpoint()
        #创建一个场景以渲染网格，设置渲染图像的大小为512。
        render_mesh = create_scene([m.eval()], sz=512)

        #在第一次迭代时，克隆渲染的网格并计算法线和切线。
        if it == 0:
            base_mesh = render_mesh.clone()
            base_mesh = mesh.auto_normals(base_mesh)
            base_mesh = mesh.compute_tangents(base_mesh)
        render_mesh = mesh.auto_normals(render_mesh)
        render_mesh = mesh.compute_tangents(render_mesh)

        # Logging mesh
        #每隔一定的迭代次数（由cfg.log_interval指定）进行日志记录。
        if it % cfg.log_interval == 0:
            with torch.no_grad():
                params = get_camera_params(
                    cfg.log_elev,
                    rot_ang,
                    cfg.log_dist,
                    cfg.log_res,
                    cfg.log_fov,
                )
                rot_ang += 1
                log_mesh = mesh.unit_size(render_mesh.eval(params))
                log_image = render.render_mesh(
                    glctx,
                    log_mesh,
                    params['mvp'],
                    params['campos'],
                    params['lightpos'],
                    cfg.log_light_power,
                    cfg.log_res,
                    1,
                    background=torch.ones(1, cfg.log_res, cfg.log_res, 3).to(device)
                )

                log_image = video.ready_image(log_image)
                logger.add_mesh('predicted_mesh', vertices=log_mesh.v_pos.unsqueeze(0), faces=log_mesh.t_pos_idx.unsqueeze(0), global_step=it)
        
        if cfg.adapt_dist and it > 0:
            with torch.no_grad():
                v_pos = m.v_pos.clone() #克隆当前网格的顶点位置。
                vmin = v_pos.amin(dim=0) #计算顶点位置的最小值和最大值。
                vmax = v_pos.amax(dim=0)
                v_pos -= (vmin + vmax) / 2 #将顶点位置中心化。
                mult = torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().cpu() #计算顶点位置的缩放因子。
                cams.dataset.dist_min = cfg.dist_min * mult #根据缩放因子调整相机的最小和最大距离。
                cams.dataset.dist_max = cfg.dist_max * mult

        params_camera = next(iter(cams)) #从相机数据加载器中获取下一个相机参数。
        # print("",params_camera)
        for key in params_camera:
            params_camera[key] = params_camera[key].to(device)
        
        #breakpoint()
        final_mesh = render_mesh.eval(params_camera) #根据相机参数评估最终的渲染网格。

        # breakpoint()
        #渲染最终网格并生成训练图像。
        train_render = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs']
        ).permute(0, 3, 1, 2) #train_render.shape=torch.Size([1, 3, 512, 512])
        # breakpoint()
        print("train_render.shape= ", train_render.shape)
        sds_loss = torch.tensor(0.0, device=device)  # 初始化为
        train_render_bf16 = train_render.to(torch.bfloat16)
        latents = pipe.vae.encode(train_render_bf16).latent_dist.sample() # torch.Size([1, 16, 28, 28])
        latents = shift_scale_latents(latents,vae_scale_factor)
        print("latents.shape: ", latents.shape)
        # breakpoint()
        with torch.no_grad():
            # breakpoint()       
            # sample noise, call unet, get target
            # unet.to(accelerator.device,dtype=unet_weight_dtype) # move to device because unet is not prepared by accelerator #此处报oom
            noise_pred, target, sigmas, weighting = get_noise_pred_and_target_Rev(
                cfg,# 训练参数配置
                accelerator, # 加速器对象
                noise_scheduler,# 噪声调度器
                latents,  # 潜在空间表示 #torch.Size([1, 16, 28, 28])
                text_encoder_conds, # 文本编码器条件
                unet,  # UNet模型
                weight_dtype, #权重数据类型 torch.float16 or torch.bfloat16
                train_unet, # 是否训练UNet
            )
            if noise_pred is None:
                raise ValueError("noise_pred is not assigned!")
        # # 确保 noise_pred 和 target 需要梯度
        # # noise_pred.requires_grad_(True)
        # # target.requires_grad_(True)
        # # breakpoint()
        # # sds_loss = torch.nn.functional.mse_loss(noise_pred, target, reduction='none').mean() #noise_pred.shape=target.shap=torch.Size([1, 16, 28, 28])
        # weighting_scheme = "sigma_sort"  # 或 "cosmap"
        # # 计算 weighting
        # weighting = compute_loss_weighting_for_sd3(weighting_scheme, sigmas) #{weighting.dtype}=torch.float32
        # weighting = weighting.to(torch.bfloat16)#{weighting.dtype}=torch.float16
        # print("weighting = ",weighting)
        # grad = weighting*(noise_pred - target) 
        # # grad = (noise_pred - target) 
        # grad = torch.nan_to_num(grad) #防止noise为none，极大值，极小值导致训练崩掉
        # # breakpoint()
        # #这样手动指定梯度，可以梯度回传的时候有梯度了
        # # sds_loss = SpecifyGradient.apply(latents, grad) #sds_loss=tensor([1.], device='cuda:0', grad_fn=<SpecifyGradientBackward>) loss.requires_grad=True  original_sds loss
        # targets = (latents - grad).detach()
        # sds_loss = 0.5 * torch.nn.functional.mse_loss(latents, targets, reduction='sum') / latents.shape[0] #tensor(5100.5249, device='cuda:0', grad_fn=<DivBackward0>)  {sds_loss.dtype}=float32
        # logger.add_scalar('sds_loss', sds_loss, global_step=it)
        # print("sds_loss.requires_grad= ",sds_loss.requires_grad) # True
        # breakpoint()

        #####################modify follow sd3 flowmatching
        grad = torch.nan_to_num(noise_pred)
        target1 = (grad).detach()
        print("weighting = ",weighting) #weighting =  tensor([0.7608], device='cuda:0')
        # breakpoint()
        sds_loss =  torch.nn.functional.mse_loss(target-latents, target1, reduction="mean") / latents.shape[0] #RFDS_Rev中，sds_loss中没有weightiing；分母应该是多少？ latents.shape[0]=1

        train_render = resize(train_render, out_shape=(224, 224), interp_method=resize_method) #torch.Size([25, 3, 224, 224])
        
        # breakpoint()
        #渲染最终网格并返回光栅化映射。
        train_rast_map = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs'],
            return_rast_map=True
        )

        if it == 0:
            params_camera = next(iter(cams))
            for key in params_camera:
                params_camera[key] = params_camera[key].to(device)
        # breakpoint()
        #渲染基础网格并生成基础图像。
        base_render = render.render_mesh(
            glctx,
            base_mesh.eval(params_camera),
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs'],
        ).permute(0, 3, 1, 2)
        base_render = resize(base_render, out_shape=(224, 224), interp_method=resize_method)
        
        if it % cfg.log_interval_im == 0:
            log_idx = torch.randperm(cfg.batch_size)[:5]#随机选择5个索引以进行日志记录。
            s_log = train_render[log_idx, :, :, :] #从训练渲染中提取选定的图像。
            s_log = torchvision.utils.make_grid(s_log) #将选定的图像合并为一个网格。
            ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()#将图像数据转换为NumPy数组，以便保存。
            im = Image.fromarray(ndarr) #从NumPy数组创建图像对象。
            im.save(str(output_path / 'images' / f'epoch_{it}.png')) #将图像保存到指定路径

            obj.write_obj(
                str(output_path / 'mesh_final'),
                m.eval()
            )
        
        optimizer.zero_grad()
        # breakpoint()
        #目的：想渲染正视角的时候，进行l2 loss，其他视角则进行原有的loss

        # 1. 获取索引并创建mask
        # index = params_camera['index']
        # mask = torch.eq(index % 2, 0)  # mask为True的位置是需要多计算loss的样本
                
        # breakpoint()
        # CLIP similarity losses
        normalized_clip_render = (train_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None] #对形变mesh渲染图片进行归一化
        image_embeds = model.encode_image( #形变mesh渲染图片进行clip编码为image embedding
            normalized_clip_render
        )
        with torch.no_grad():
            normalized_base_render = (base_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
            base_embeds = model.encode_image(normalized_base_render)
        
        orig_image_embeds = image_embeds.clone() / image_embeds.norm(dim=1, keepdim=True)
        delta_image_embeds = image_embeds - base_embeds #计算当前图像嵌入与基础图像嵌入之间的差异。
        delta_image_embeds = delta_image_embeds / delta_image_embeds.norm(dim=1, keepdim=True)

        #计算当前图像嵌入与目标文本嵌入之间的余弦相似度损失。
        clip_loss = cosine_avg(orig_image_embeds, target_text_embeds)  #对应公式（2）
        #计算差异图像嵌入与目标文本差异嵌入之间的余弦相似度损失。
        delta_clip_loss = cosine_avg(delta_image_embeds, delta_text_embeds) #对应公式（3）
        logger.add_scalar('clip_loss', clip_loss, global_step=it)
        logger.add_scalar('delta_clip_loss', delta_clip_loss, global_step=it)
        # breakpoint()

        # Jacobian regularization
        r_loss = (((gt_jacobians) - torch.eye(3, 3, device=device)) ** 2).mean()
        logger.add_scalar('jacobian_regularization', r_loss, global_step=it)
        # breakpoint()

        # Consistency loss
        # Get mapping from vertex to pixels
        curr_vp_map = get_vp_map(final_mesh.v_pos, params_camera['mvp'], 224) #获取当前网格顶点到像素的映射
        for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(cfg.batch_size, -1)):
            u_faces = rast_faces.unique().long()[1:] - 1
            t = torch.arange(len(final_mesh.v_pos), device=device)
            u_ret = torch.cat([t, final_mesh.t_pos_idx[u_faces].flatten()]).unique(return_counts=True)
            non_verts = u_ret[0][u_ret[1] < 2]
            curr_vp_map[idx][non_verts] = torch.tensor([224, 224], device=device)
        
        # Get mapping from vertex to patch
        med = (fe.old_stride - 1) / 2
        curr_vp_map[curr_vp_map < med] = med
        curr_vp_map[(curr_vp_map > 224 - fe.old_stride) & (curr_vp_map < 224)] = 223 - med
        curr_patch_map = ((curr_vp_map - med) / fe.new_stride).round()
        flat_patch_map = curr_patch_map[..., 0] * (((224 - fe.old_stride) / fe.new_stride) + 1) + curr_patch_map[..., 1]
        
        # Deep features
        patch_feats = fe(normalized_clip_render)
        flat_patch_map[flat_patch_map > patch_feats[0].shape[-1] - 1] = patch_feats[0].shape[-1]
        flat_patch_map = flat_patch_map.long()[:, None, :].repeat(1, patch_feats[0].shape[1], 1)

        deep_feats = patch_feats[cfg.consistency_vit_layer]
        deep_feats = torch.nn.functional.pad(deep_feats, (0, 1))
        deep_feats = torch.gather(deep_feats, dim=2, index=flat_patch_map)
        deep_feats = torch.nn.functional.normalize(deep_feats, dim=1, eps=1e-6)

        elev_d = torch.cdist(params_camera['elev'].unsqueeze(1), params_camera['elev'].unsqueeze(1)).abs() < torch.deg2rad(torch.tensor(cfg.consistency_elev_filter))
        azim_d = torch.cdist(params_camera['azim'].unsqueeze(1), params_camera['azim'].unsqueeze(1)).abs() < torch.deg2rad(torch.tensor(cfg.consistency_azim_filter))

        cosines = torch.einsum('ijk, lkj -> ilk', deep_feats, deep_feats.permute(0, 2, 1))
        cosines = (cosines * azim_d.unsqueeze(-1) * elev_d.unsqueeze(-1)).permute(2, 0, 1).triu(1)
        consistency_loss = cosines[cosines != 0].mean()
        logger.add_scalar('consistency_loss', consistency_loss, global_step=it)
        # breakpoint()
        #only sds_loss setting
        # total_loss = sds_loss.mean() #有梯度了
        # print(f"total_loss 数据类型: {total_loss.dtype}")

        #sds_loss+clip相关loss的setting
        sds_loss = sds_loss.mean()
        total_loss = cfg.flux_weight * sds_loss + \
            cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss + cfg.regularize_jacobians_weight * r_loss
        # total_loss = cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss + \
        #     cfg.regularize_jacobians_weight * r_loss
        # breakpoint()
        # total_loss = cfg.sds_weight * l_guidance + \
        #     cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss + \
        #     cfg.regularize_jacobians_weight * r_loss + cfg.l2_weight*l2_loss 

        # 确保 total_loss 是标量
        # # breakpoint()
        # # total_loss_scaler = total_loss.mean().item()  # 计算平均损失并转换为标量 #type(total_loss)=<class 'torch.Tensor'>, type(total_loss_scaler)=float
        # logger.add_scalar('total_loss', total_loss_scaler, global_step=it)
        # if best_losses['total'] > total_loss_scaler: #如果当前损失优于之前的最佳损失，则更新最佳损失病保存当前网格
        #     best_losses['total'] = total_loss.detach()
        #     obj.write_obj(
        #         str(output_path / 'mesh_best_total'),
        #         m.eval()
        #     )

        if best_losses['total'] > total_loss: #如果当前损失优于之前的最佳损失，则更新最佳损失病保存当前网格
            best_losses['total'] = total_loss.detach()
            obj.write_obj(
                str(output_path / 'mesh_best_total'),
                m.eval()
            )
        logger.add_scalar('total_loss', total_loss, global_step=it)
        # breakpoint()
        if best_losses['CLIP'] > clip_loss: #如果当前CLIP损失优于之前的最佳损失，则更新最佳损失并保存前网格
            best_losses['CLIP'] = clip_loss.detach()
            obj.write_obj(
                str(output_path / 'mesh_best_clip'),
                m.eval()
            )
        # breakpoint()
        # total_loss = total_loss.mean()
        total_loss.backward() #执行反向传播，计算梯度 tensor 才可以求梯度
        print("gt_jacobians.grad:", gt_jacobians.grad) #torch.Size([1, 20868, 3, 3])
        # breakpoint()
        optimizer.step() #更新优化器的参数
        # t_loop.set_description(f'CLIP Loss = {clip_loss.item()}, SDS Loss = {l_guidance.item()},Total Loss = {total_loss.item()}')
        # print(f"CLIP Loss: {clip_loss}, SDS Loss: {cfg.sds_weight * l_guidance}, Jacobian Loss: {r_loss},L2 Loss: {l2_loss}, Consistency Loss: {consistency_loss}, Total Loss = {total_loss}")
        print(f"CLIP Loss: {clip_loss}, SDS Loss: {sds_loss}, Jacobian Loss: {r_loss} Total Loss = {total_loss}")
        # t_loop.set_description(f'CLIP Loss = {clip_loss.item()},Total Loss = {total_loss.item()}')
        
        # 在每个epoch结束时清理缓存
        torch.cuda.empty_cache()
    
    video.close() #关闭视屏记录
    obj.write_obj( #将最终网格保存为OBJ文件。
        str(output_path / 'mesh_final'),
        m.eval()
    )
    
    return
