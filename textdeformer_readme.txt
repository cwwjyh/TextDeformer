1. 所有的配置文件都在 ./config   目录下，每一个配置文件对应一个要编辑的mesh文件，输入，输出，以及各种超参数的配置
2. TextDeforme/loop_nine_grid_flux.py 将渲染的图片拼接为9宫格，然后送给flux加噪，去噪。对应的camera.py 为：TextDeformer/utilities/camera_nine_grid_flux.py； 对应的config文件有很多，比如：TextDeformer/config/donkey2horse_flux_nine_grid_1000_fix_black.yml
3. TextDeformer/loop_one_image_flux.py 只渲染一张图片，batch_size=1, 对应的camera.py 为：TextDeformer/utilities/camera_one_image_flux.py: 对应的config文件有很多，比如：TextDeformer/config/donkey2horse_flux_one_image_5000_fix-hyp_bs1.yml
4. TextDeformer/loop_mv_imag_flx_bs=16.py，可以渲染16张图片，batch_size=16, 对应的camera.py 为：TextDeformer/utilities/camera_mv_img_flux_bs=16.py
5. TextDeformer/loop.py， 渲染20张图片，前4张使用l2 loss约束，后16张图片使用原始的clip loss，对应的camera.py 为：TextDeformer/utilities/camera.py， 对应的config文件有很多，比如：TextDeformer/config/donkey2dog_mv_ref_1000epo_l2_5.yml
6. ./loop_two_sd.py和 ./loop_two_sd.py即用传统的sds loss的结果
7， 注：要使用对应的loop.py 要对应为对应的camera.py。 eg：loop_nine_grid_flux.py 对应camera_nine_grid_flux.py，运行的时候该设定的时候需要修改文件名：loop_nine_grid_flux.py--> loop.py and camera_nine_grid_flux.py-->came
