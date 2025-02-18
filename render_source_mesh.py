import os
import yaml
import torch
import pathlib
import pymeshlab
import argparse
import numpy as np
import kornia
import yaml, logging
from easydict import EasyDict
from PIL import Image

from NeuralJacobianFields import SourceMesh

import nvdiffrast.torch as dr
from nvdiffmodeling.src     import obj
from nvdiffmodeling.src     import util
from nvdiffmodeling.src     import mesh
from nvdiffmodeling.src     import render
from nvdiffmodeling.src     import texture
from nvdiffmodeling.src     import regularizer

from utilities.video import Video
from utilities.helpers import cosine_avg, create_scene, get_vp_map
from utilities.camera import CameraBatch, get_camera_params
from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3

def make_grid_with_cat(images, nrow):
    """
    使用 torch.cat 拼接图像，保留梯度信息。
    :param images: 输入图像张量，形状为 [N, C, H, W]
    :param nrow: 每行的图像数量
    :return: 拼接后的图像张量，形状为 [C, H * nrows, W * ncols]
    """
    N, C, H, W = images.shape
    ncols = (N + nrow - 1) // nrow  # 计算列数

    # 按行拼接图像
    rows = []
    for i in range(ncols):
        row_images = images[i * nrow:(i + 1) * nrow]
        row = torch.cat([img for img in row_images], dim=-1)  # 按列拼接
        rows.append(row)

    # 按行拼接所有行
    grid = torch.cat(rows, dim=-2)
    return grid


def main(cfg):
    output_path = pathlib.Path(cfg['output_path'])
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / 'config.yml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    cfg = EasyDict(cfg)

    device = torch.device(f'cuda:{cfg.gpu}')
    torch.cuda.set_device(device)

    if cfg.resize_method == 'cubic':
        resize_method = cubic
    elif cfg.resize_method == 'linear':
        resize_method = linear
    elif cfg.resize_method == 'lanczos2':
        resize_method = lanczos2
    elif cfg.resize_method == 'lanczos3':
        resize_method = lanczos3

    # GL Context
    glctx = dr.RasterizeCudaContext() #创建一个 CUDA 上下文，用于图形渲染。

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

    # Determine guidance prompt
    # prompt = f"a photo of {cfg.text_prompt}" #before writting
    prompt = f"a photo of sks {cfg.text_prompt}"
    print(f"Guidance Prompt: {str(prompt)}")
    print("target_prompt:", cfg.target_prompt)

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

    jacobian_source = SourceMesh.SourceMesh(0, str(output_path / 'tmp' / 'mesh.obj'), {}, 1, ttype=torch.float)
    if len(list((output_path / 'tmp').glob('*.npz'))) > 0:
        logging.warn(f'Using existing Jacobian .npz files in {str(output_path)}/tmp/ ! Please check if this is intentional.') 
    jacobian_source.load()
    jacobian_source.to(device)

    with torch.no_grad():
        #从加载的网格顶点计算雅可比矩阵
        gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
    gt_jacobians.requires_grad_(False) #设置雅可比矩阵以计算梯度。

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
    render_mesh = create_scene([m.eval()], sz=512)

    render_mesh = mesh.auto_normals(render_mesh)
    render_mesh = mesh.compute_tangents(render_mesh)


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
        
    final_mesh = render_mesh.eval(params_camera) #根据相机参数评估最终的渲染网格。
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
    train_render = resize(train_render, out_shape=(224, 224), interp_method=resize_method)
    # 使用 torch.cat 拼接函数
    s_log = train_render[:9, :, :, :]
    # print("s_log.requires_grad=", s_log.requires_grad)  # 打印 True
    # print("s_log:", s_log.grad)#None

    s_log_grid = make_grid_with_cat(s_log, nrow=3)
    # from torchvision.utils import save_image
    ndarr = s_log_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(str(output_path / f'source_mesh.png'))

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
    main(cfg)