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
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/donkey2giraffe_flux.yml
```