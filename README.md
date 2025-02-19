## Render Source Mesh
```bash
CUDA_VISIBLE_DEVICES=0 python render_source_mesh.py --config config/donkey2camel_flux.yml
```

## Query GPT
```bash
export no_proxy="ivolces.com,clear.ml,kubebrain,kubebrain.com,svc,basemind.local,basemind.com,basemind.ml,127.0.0.1,10.0.0.0/8,localhost,svc,stepfun-inc.com"
CUDA_VISIBLE_DEVICES=0 python gpt-4o.py --config config/donkey2camel_flux.yml
```