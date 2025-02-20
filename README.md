## Rotate Mesh
```bash
python mesh_rotate.py --name max_planck --angle 90 --axis y  # left = +; right = -
```

## Render Source Mesh
```bash
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/donkey2camel_flux.yml
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/donkey2horse_flux.yml
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/donkey2giraffe_flux.yml
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/cow2giraffe_flux.yml
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/guitar2banjo_flux.yml
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/alien2scary_flux.yml
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/shoe1_highheel_flux.yml
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/shoe2_armyboot_flux.yml
```

## Query GPT
```bash
export no_proxy="ivolces.com,clear.ml,kubebrain,kubebrain.com,svc,basemind.local,basemind.com,basemind.ml,127.0.0.1,10.0.0.0/8,localhost,svc,stepfun-inc.com"
CUDA_VISIBLE_DEVICES=0 python gpt-4o.py --config config/donkey2camel_flux.yml
CUDA_VISIBLE_DEVICES=0 python gpt-4o.py --config config/donkey2horse_flux.yml
CUDA_VISIBLE_DEVICES=0 python gpt-4o.py --config config/donkey2giraffe_flux.yml
```

## Flux
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2camel_flux.yml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/donkey2horse_flux.yml
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/donkey2giraffe_flux.yml
CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/shoe2_armyboot_flux.yml
CUDA_VISIBLE_DEVICES=4 python main.py --config config/shoe1_highheel_flux.yml
CUDA_VISIBLE_DEVICES=5 python main.py --config config/guitar2banjo_flux.yml
CUDA_VISIBLE_DEVICES=6 python main.py --config config/vase1_royalgoblet_flux.yml
CUDA_VISIBLE_DEVICES=7 python main.py --config config/vase2_heartvase_flux.yml

CUDA_VISIBLE_DEVICES=0 python main.py --config config/eiffel_skyscraper_flux.yml
CUDA_VISIBLE_DEVICES=0 python main.py --config config/chair1_gamingchair_flux.yml


```

## Flux infer
```bash
CUDA_VISIBLE_DEVICES=0 python infer_flux_lora.py --config ./config/bird_bat_flux.yml
CUDA_VISIBLE_DEVICES=0 python infer_flux_lora.py --config ./config/shoe2_armyboot_flux.yml

CUDA_VISIBLE_DEVICES=0 python infer_flux_lora.py --config config/donkey2camel_flux.yml
CUDA_VISIBLE_DEVICES=0 python infer_flux_lora.py --config config/donkey2horse_flux.yml
CUDA_VISIBLE_DEVICES=0 python infer_flux_lora.py --config config/donkey2giraffe_flux.yml
```