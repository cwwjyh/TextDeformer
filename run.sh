# python main.py --config config/vase2royal_goblet.yml
# python main.py --config config/vase2royal_goblet_a_0.yml
# python main.py --config config/vase2royal_goblet_a_1.yml
# python main.py --config config/vase2royal_goblet_a_5.yml
# python main.py --config config/vase2royal_goblet_a_10.yml

# python main.py --config config/example_config_donkey2zebra.yml #有点扁
# python main.py --config config/example_config_donkey2zebra_a_5.yml #效果还可以
# python main.py --config config/example_config_donkey2zebra_a_10.yml  #效果还可以
# python main.py --config config/example_config_donkey2zebra_a_5000iter.yml #效果很差，前后腿不一样长
# python main.py --config config/example_config_donkey2zebra_clip_0.yml #效果很差，前腿变特别短
# python main.py --config config/example_config_donkey2zebra_clip_2.yml #效果一般，后腿又一个很长
# python main.py --config config/example_config_donkey2zebra_deta_clip_2.yml #效果不好
# python main.py --config config/example_config_donkey2zebra_deta_clip_0.yml #变成梅花鹿了


# python main.py --config config/example_config_donkey2cow_delta_clip_0.yml
# python main.py --config config/example_config_pig2dog_delta_clip_0.yml



# python main.py --config config/car2bus_sw_100_sds.yml
# python main.py --config config/frog2teddy_bear_sw_100_sds.yml
# python main.py --config config/laptop2book_sw_100_sds.yml


# python main.py --config config/pig_sw_100_sds_longtext.yml
# python main.py --config config/pigear_sw_100_sds_longtext.yml

# python main.py --config config/pigleg_sw_100_sds_longtext.yml
# python main.py --config config/pigear1_sw_100_sds_longtext.yml  # a long ears pig  

# python main.py --config config/pigear2_sw_100_sds_longtext.yml #A pig with long ears.


# python main.py --config config/pigear3_sw_1_sds_longtext.yml #a long ears pig
# python main.py --config config/pigear3_sw_25_sds_longtext.yml #a long ears pig
# python main.py --config config/pigear3_sw_50_sds_longtext.yml #a long ears pig
# python main.py --config config/pigear3_sw_10_sds_longtext.yml #a long ears pig

# python main.py --config config/pigear3_sw_1_sds_longtext_500epo_clip.yml #a long ears pig
# python main.py --config config/pigear3_sw_10_sds_longtext_500epo_clip.yml #a long ears pig
# python main.py --config config/pigear3_sw_25_sds_longtext_500epo_clip.yml #a long ears pig

#batch_size=16的时候，显存74G，感觉还可以调整到bz=18
# python main.py --config ./config/donkey2horse_flux.yml
# python main.py --config ./config/donkey2horse_flux_1000epo.yml
# python main.py --config ./config/donkey2horse_flux_2000epo.yml
# python main.py --config ./config/donkey2horse_flux_2500epo.yml
# python main.py --config ./config/donkey2horse_flux_nine_grid_3000.yml
# python main.py --config ./config/donkey2horse_flux_nine_grid_2000.yml
# python main.py --config ./config/donkey2horse_flux_nine_grid_500000.yml
# python main.py --config ./config/donkey2horse_flux_5000000epo.yml


# python main.py --config ./config/donkey2horse_flux_nine_grid_3000_pk.yml
# python main.py --config ./config/donkey2horse_flux_nine_grid_2000_pk.yml

# python main.py --config ./config/donkey2horse_flux_nine_grid_1000.yml
# python main.py --config ./config/donkey2horse_flux_nine_grid_1000_black.yml

# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_2000.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_black.yml

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_500000_bk.yml

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_3000.yml



# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_2000_fix_black.yml

# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_1000_fix.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_1000_fix_black.yml




# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_1000_fix-hyp.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_5000_fix-hyp_bs13.yml

# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_3000_fix-hyp_bs13.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2horse_flux_nine_grid_4000_fix-hyp_bs13.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux_nine_grid_5000000_fix-hyp_bs13.yml

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_only_flux_one_image_2000_fix-hyp_bs1_test_fix_camera_add_weight.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_only_flux_one_image_3000_fix-hyp_bs1_test_fix_camera_add_weight.yml

#add classifeir guidance like sds loss guidance_sclae=100
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_only_flux_one_image_2000_fix-hyp_bs1_test_fix_camera_add_weight_cfg.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_only_flux_one_image_3000_fix-hyp_bs1_test_fix_camera_add_weight_cfg.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_only_flux_one_image_999000_fix-hyp_bs1_test_fix_camera_add_weight.yml

#guidance_scale=10
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_only_flux_one_image_2000_fix-hyp_bs1_test_fix_camera_add_weight_10cfg.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_only_flux_one_image_3000_fix-hyp_bs1_test_fix_camera_add_weight_10cfg.yml

#add l2 loss, consistency_loss对所有train_render进行的， 测试l2 loss前的权重
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_mv_ref_1000epo_l2_5.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_mv_ref_1000epo_l2_10.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_mv_ref_1000epo_l2_50.yml

#换case测试
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2dog_mv_ref_1000epo_l2_5.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2deer_mv_ref_1000epo_l2_5.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/cow2deer_mv_ref_1000epo_l2_5.yml
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2cow_mv_ref_1000epo_l2_5.yml #正在跑

# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2dog_mv_ref_1000epo_l2_10.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2deer_mv_ref_1000epo_l2_10.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/cow2deer_mv_ref_1000epo_l2_10.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2cow_mv_ref_1000epo_l2_10.yml #正在跑


# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2dog_mv_ref_1000epo_l2_20.yml
# CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2deer_mv_ref_1000epo_l2_15.yml  #
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2deer_mv_ref_1000epo_l2_20.yml  #
# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2cow_mv_ref_1000epo_l2_20.yml 

#利用objaverse渲染的9个视角作为gt，使用l2 loss
#camera 没有与gt camera对齐
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_1000epo_l2_10_bs25.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_1000epo_l2_15_bs25.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_1000epo_l2_20_bs25.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_2000epo_l2_10_bs25.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_2000epo_l2_15_bs25.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_2000epo_l2_20_bs25.yml




# CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2donkey_gt_mv_ref_1000epo_l2_10_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2donkey_gt_mv_ref_1000epo_l2_15_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2donkey_gt_mv_ref_1000epo_l2_20_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2donkey_gt_mv_ref_1000epo_l2_25_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2donkey_gt_mv_ref_2000epo_l2_10_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2donkey_gt_mv_ref_2000epo_l2_15_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2donkey_gt_mv_ref_2000epo_l2_20_25bs.yml


CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2dog_gt_mv_ref_1000epo_l2_10_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2dog_gt_mv_ref_1000epo_l2_15_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2dog_gt_mv_ref_1000epo_l2_20_25bs.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2dog_gt_mv_ref_2000epo_l2_10_25bs.yml


#camera 与gt camera对齐
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_1000epo_l2_10_bs25_re.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_1000epo_l2_15_bs25_re.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_1000epo_l2_20_bs25_re.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_2000epo_l2_10_bs25_re.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_2000epo_l2_15_bs25_re.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/pig2pig_gt_mv_ref_2000epo_l2_20_bs25_re.yml


#增加spot case
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/spot2cow_gt_mv_ref_1000epo_l2_10_25bs.yml


./gui.sh --listen 127.0.0.1 --server_port 7860 --share --headless 