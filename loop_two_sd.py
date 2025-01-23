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
from src.guidance.stable_diffusion import StableDiffusionGuidance

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

    #TODO之前为cww add
    #  Initialize guidance
    guidance = StableDiffusionGuidance(
        device=torch.device("cuda"),
        lora_path=cfg.lora_dir,
        lora_scale=cfg.lora_scale,
        grad_clamp_val=cfg.clamp_val,
    )
    print(f"Loaded guidance: {type(guidance)}")

    # Determine guidance prompt
    # prompt = f"a photo of {cfg.text_prompt}" #before writting
    prompt = f"a photo of sks {cfg.text_prompt}"
    print(f"Guidance Prompt: {str(prompt)}")
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
    # cams_data = CameraBatch( #初始化相机 有batch_size个相机参数
    #     cfg.train_res,
    #     [cfg.dist_min, cfg.dist_max],
    #     [cfg.azim_min, cfg.azim_max],
    #     [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
    #     [cfg.fov_min, cfg.fov_max],
    #     cfg.aug_loc,
    #     cfg.aug_light,
    #     cfg.aug_bkg,
    #     cfg.batch_size,
    #     rand_solid=True
    # )
    # cams = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True) #创建相机数据加载器。
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

    for it in t_loop:
          # 根据当前迭代次数设置batch_size
        if it >= 500 :
            batch_size = 25  # 前500次迭代时的batch_size
                # 初始化相机数据
            cams_data = CameraBatch(  # 使用动态的cfg.batch_size
                cfg.train_res,
                [cfg.dist_min, cfg.dist_max],
                [cfg.azim_min, cfg.azim_max],
                [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
                [cfg.fov_min, cfg.fov_max],
                cfg.aug_loc,
                cfg.aug_light,
                cfg.aug_bkg,
                batch_size,  # 使用动态的batch_size
                rand_solid=True
            )
        else:
            batch_size = 1  # 超过500次迭代时的batch_size
            consistency_loss = 0

            # 初始化相机数据            
                # 设置相机参数以获取正视角
            elev_alpha = 0  # 俯仰角设置为0度
            azim_min = 0    # 方位角最小值设置为0度
            azim_max = 0    # 方位角最大值设置为0度
            cams_data = CameraBatch(  # 使用动态的cfg.batch_size
                cfg.train_res,
                [cfg.dist_min, cfg.dist_max],
                [azim_min, azim_max],
                [elev_alpha, cfg.elev_beta, cfg.elev_max],
                [cfg.fov_min, cfg.fov_max],
                cfg.aug_loc,
                cfg.aug_light,
                cfg.aug_bkg,
                batch_size,  # 使用动态的batch_size
                rand_solid=True
            )

        # # 初始化相机数据
        # cams_data = CameraBatch(  # 使用动态的cfg.batch_size
        #     cfg.train_res,
        #     [cfg.dist_min, cfg.dist_max],
        #     [cfg.azim_min, cfg.azim_max],
        #     [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
        #     [cfg.fov_min, cfg.fov_max],
        #     cfg.aug_loc,
        #     cfg.aug_light,
        #     cfg.aug_bkg,
        #     batch_size,  # 使用动态的batch_size
        #     rand_solid=True
        # )
        cams = torch.utils.data.DataLoader(cams_data, batch_size, num_workers=0, pin_memory=True)  # 创建相机数据加载器


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
        #每隔一定的迭代次数（由cfg.log_interval指定）进行日���记录。
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
        ).permute(0, 3, 1, 2)

        train_render = resize(train_render, out_shape=(224, 224), interp_method=resize_method) #torch.Size([1, 3, 224, 224])
        
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
            # log_idx = torch.randperm(cfg.batch_size)[:5]#随机选择5个索引以进行日志记录。
            log_idx = torch.randperm(batch_size)[:5]#随机选择5个索引以进行日志记录。
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

        #compute sds loss(cww add)
        l_guidance = torch.tensor(0.0).to(device) #初始化引导损失为0。
      
        if it < 500:
            #计算引导损失，使用当前的训练图像和目标提示。
            l_guidance = guidance(
                prompt, #换成目标prompt
                image=train_render, #train_render.shape = torch.Size([1, 3, 224, 224]) #batch_size=1
                cfg_scale=cfg.cfg_scale,
            )
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
        
        if it > 500:
            # Consistency loss
            # Get mapping from vertex to pixels
            curr_vp_map = get_vp_map(final_mesh.v_pos, params_camera['mvp'], 224) #获取当前网格顶点到像素的映射
            for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(batch_size, -1)):
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
        # # breakpoint()
        total_loss = cfg.sds_weight * l_guidance + \
            cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss + \
            cfg.regularize_jacobians_weight * r_loss - cfg.consistency_loss_weight * consistency_loss
        # breakpoint()
        # total_loss = cfg.sds_weight * l_guidance + \
        #     cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss + \
        #     cfg.regularize_jacobians_weight * r_loss 
        logger.add_scalar('total_loss', total_loss, global_step=it)

        if best_losses['total'] > total_loss: #如果当前损失优于之前的最佳损失，则更新最佳损失病保存当前网格
            best_losses['total'] = total_loss.detach()
            obj.write_obj(
                str(output_path / 'mesh_best_total'),
                m.eval()
            )
        if best_losses['CLIP'] > clip_loss: #如果当前CLIP损失优于之前的最佳损失，则更新最佳损失病保存当前网格
            best_losses['CLIP'] = clip_loss.detach()
            obj.write_obj(
                str(output_path / 'mesh_best_clip'),
                m.eval()
            )

        total_loss.backward() #执行反向传播，计算梯度
        optimizer.step() #更新优化器的参数
        # t_loop.set_description(f'CLIP Loss = {clip_loss.item()}, SDS Loss = {l_guidance.item()},Total Loss = {total_loss.item()}')
        print(f"CLIP Loss: {clip_loss}, SDS Loss: {l_guidance}, Jacobian Loss: {r_loss}, Consistency Loss: {consistency_loss}, Total Loss = {total_loss}")
        # t_loop.set_description(f'CLIP Loss = {clip_loss.item()},Total Loss = {total_loss.item()}')
    
    video.close() #关闭视屏记录
    obj.write_obj( #将最终网格保存为OBJ文件。
        str(output_path / 'mesh_final'),
        m.eval()
    )
    
    return
