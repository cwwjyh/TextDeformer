# python main.py --config config/example_config_donkey2cow_a_5.yml 
# python main.py --config config/example_config_donkey2cow_a_10.yml
# python main.py --config config/example_config_donkey2cow_a_1.yml
# python main.py --config config/example_config_donkey2cow_5000itera.yml #缺一个后腿，效果最差，所以epoch还是设置为2500

# python main.py --config config/example_config_spot2Ladybug.yml 
# python main.py --config config/example_config_spot2snail.yml 
# python main.py --config config/example_config_spot2turtle.yml 


# python main.py --config config/pigear3_sw_50_sds_longtext_500epo_clip.yml #a long ears pig
# python main.py --config config/pigear3_sw_100_sds_longtext_500epo_clip.yml #a long ears pig

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_3000.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_3000_fix.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix-hyp_test.yml 

# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_500000_bk.yml



#add l2 loss, consistency_loss对所有train_render进行的， 测试l2 loss前的权重
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_mv_ref_1000epo_l2_5_cl.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_mv_ref_1000epo_l2_10_cl.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_mv_ref_1000epo_l2_50_cl.yml


#测试l2前权重为15的时候的效果，其中donkey2dog的效果比贴合所给的图片s
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2dog_mv_ref_1000epo_l2_15.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2deer_mv_ref_1000epo_l2_15.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2dog_mv_ref_1000epo_l2_15.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/cow2deer_mv_ref_1000epo_l2_15.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2cow_mv_ref_1000epo_l2_15.yml #正在

# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_one_image_5000_fix-hyp_bs1_test.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_one_image_5000_fix-hyp_bs1_zhanka.yml


# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_one_image_5000_fix-hyp_bs1_test2_modify.yml

# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_one_image_5000_fix-hyp_bs1_test3.yml

# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/spot2cow_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2dog_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev_zhanka.yml

####################
#lora_weight = "/data/caiweiwei/kohya_ss/outputs/mesh_nine_grid_672_clean_prompt_re/"
#lora_name = "mesh_nine_grid_672_clean_prompt_epo-step00005000.safetensors" 下跑的结果
# python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev_relora.yml #over,效果最好
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2camel_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/spot2dachshund_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev.yml #over,几乎没有变化， 雅可比权重=0.5



# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/head2Albert_Einstein_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev.yml #over,效果不明显
# CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/head2Obama_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev.yml #over,效果不明显
# CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/head2Bust_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev.yml #over,效果不明显
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev_relora.yml #over 效果不明显
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2camel_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev1.yml ##epoch=2500, clip_weight=2


# #修改flux_weight看结果怎么样
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev_relora_2.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev_relora_3.yml #over, sds_loss在震荡下降，效果更贴近source mesh



# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/spot2dachshund_flux_nine_grid_2500_fix_hyp_test_bs_14_Rev.yml #epoch=2500, clip相关的权重=2， bs=14
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev_relora_5.yml #flux_weight=5, 更偏向于soure mesh
# CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/donkey2camel_flux_nine_grid_2000_fix-hyp_test_bs_13_Rev2 #epoch=2500, clip相关权重=2， bs=14


# #bs=17, clip相关loss=2， flux_weight=1
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/head2Bust_flux_nine_grid_2500_fix-hyp_test_bs_17_Rev.yml #runing
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/head2Albert_Einstein_flux_nine_grid_2500_fix-hyp_test_bs_17_Rev.yml #0ver, 头偏斜
# CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/head2Obama_flux_nine_grid_2500_fix-hyp_test_bs_17_Rev.yml #
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/spot2dachshund_flux_nine_grid_5000_fix_hyp_test_bs_18_Rev.yml #over， 效果不太想
# CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/donkey2camel_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev2.yml #over 效果还不错
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev2_relora.yml #runing

# #bs=17, clip相关loss=1， flux_weight=1
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/head2Bust_flux_nine_grid_2500_fix-hyp_test_bs_17_Rev1.yml #runing
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/head2Albert_Einstein_flux_nine_grid_2500_fix-hyp_test_bs_17_Rev1.yml #runing
# CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/head2Obama_flux_nine_grid_2500_fix-hyp_test_bs_17_Rev1.yml #runing
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev_relora.yml #runing


#待跑
#修改了sigmas部分，使其两次sigmas保持不变
#bs=17, clip相关loss=1， flux_weight=1
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/head2Bust_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_sigma.yml #runing
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/head2Albert_Einstein_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_sigma.yml #runing
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/head2Obama_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_sigma.yml #runing #runing
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev_relora_sigma.yml #runing
# CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/donkey2camel_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_sigma.yml 


# #bs=17, clip相关loss=2， flux_weight=1
# CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/donkey2camel_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev2_sigma.yml #
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev2_relora_sigma.yml

# ##接下来就是没有进行sigmas部分也进行了修改, l两次sigmas不一样
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/head2Obama_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_ni.yml #效果变hao，相比于18_Rev1_sigmal.yml 与17_Rev1.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/head2Bust_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_ni.yml #效果变差（鼻子歪了），相比于18_Rev1.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/head2Albert_Einstein_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_ni.yml #效果变差，相比于Rev1_sigma.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev2_relora_ni.yml #效果变差，相比于18_Rev2_relora.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2camel_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev2_ni.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix-hyp_test_bs_18_Rev_ni.yml

# #由RFDS_REV改为RFDS，并且sds前面✖️weighting，guidance_scale: 7.5
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_RFDS2_relora_7.5.yml

# #由RFDS_REV改为RFDS，并且sds前面✖️weighting，guidance_scale: 1
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_RFDS2_relora.yml
# #由RFDS_REV改为RFDS，并且sds前面✖️weighting，guidance_scale: 7.5 only sds_loss,查看sds loss到底有没有作用
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_RFDS2_relora_7.5_onlysds.yml #没有任何改变

CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/head2Albert_Einstein_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_sigma_100.yml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/head2Obama_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_sigma_100
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/head2Bust_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev1_sigma_100

##待跑
CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/spot2giraffe_flux_nine_grid_2500_fix-hyp_test_bs_18_Rev2_relora_sigma_100
spot2dachshund_flux_nine_grid_3001_fix_hyp_test_bs_18_Rev_100

