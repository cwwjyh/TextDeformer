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
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2dog_mv_ref_1000epo_l2_15.yml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2deer_mv_ref_1000epo_l2_15.yml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2dog_mv_ref_1000epo_l2_15.yml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/cow2deer_mv_ref_1000epo_l2_15.yml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/pig2cow_mv_ref_1000epo_l2_15.yml #正在