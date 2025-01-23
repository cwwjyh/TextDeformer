虚拟环境 TextDeformer按照readerme安装，除了torch按章按照 pytorch官网装的
(TextDeformer) i-caiweiwei@caiweiwei-bfkz2-229554-worker-0:/data/caiweiwei/TextDeformer-main$ python main.py --config example_config.yml
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
Aborted (core dumped)

textdeformer_new完全按照redeme安装，然后根据报错一路解决.
按照xuanyang给的解决方法解决，到后面还是报错。

textdeformer_new1完全按照redeme安装，torch按照conda install 安装：conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch 
该环境报错；segmentation fault (core dumped)


textdeformer_new2完全按照redeme安装,但是torch=2.0.1+cu117; torch.version.cuda=11.7. 报错之后按照xuanyang给的方案运行成功

加入apap中的sds loss时作出的修改：
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ pip uninstall huggingface-hub 
Found existing installation: huggingface-hub 0.26.2
Uninstalling huggingface-hub-0.26.2:
  Would remove:
    /data/anconda3/envs/textdeformer_new2/bin/huggingface-cli
    /data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/huggingface_hub-0.26.2.dist-info/*
    /data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/huggingface_hub/*
Proceed (Y/n)? y
  Successfully uninstalled huggingface-hub-0.26.2
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ pip install huggingface-hub==0.23.2

接下来报错语法错误，看报错的文件，对应到apap_new虚拟环境中对应的包是否对其。
最后发现jaxtyping版本没有与apap_new环境对其
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ python main.py --config example_config.yml 
Using /home/i-caiweiwei/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/i-caiweiwei/.cache/torch_extensions/py39_cu117/renderutils_plugin/build.ninja...
Building extension module renderutils_plugin...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module renderutils_plugin...
Traceback (most recent call last):
  File "/data/caiweiwei/TextDeformer-main/main.py", line 8, in <module>
    from loop import loop
  File "/data/caiweiwei/TextDeformer-main/loop.py", line 37, in <module>
    from src.guidance.stable_diffusion import StableDiffusionGuidance
  File "/data/caiweiwei/TextDeformer-main/src/guidance/stable_diffusion.py", line 26, in <module>
    class StableDiffusionGuidance(Guidance):
  File "/data/caiweiwei/TextDeformer-main/src/guidance/stable_diffusion.py", line 77, in StableDiffusionGuidance
    def __call__(
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/jaxtyping/_decorator.py", line 439, in jaxtyped
    full_fn = _apply_typechecker(typechecker, full_fn)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/jaxtyping/_decorator.py", line 73, in _apply_typechecker
    return typechecker(fn)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_decorators.py", line 223, in typechecked
    retval = instrument(target)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_decorators.py", line 74, in instrument
    instrumentor.visit(module_ast)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 407, in visit
    return visitor(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 621, in visit_Module
    self.generic_visit(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 513, in generic_visit
    node = super().generic_visit(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 483, in generic_visit
    value = self.visit(value)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 407, in visit
    return visitor(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 666, in visit_ClassDef
    self.generic_visit(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 513, in generic_visit
    node = super().generic_visit(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 483, in generic_visit
    value = self.visit(value)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 407, in visit
    return visitor(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 744, in visit_FunctionDef
    annotation = self._convert_annotation(deepcopy(arg.annotation))
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 605, in _convert_annotation
    new_annotation = cast(expr, AnnotationTransformer(self).visit(annotation))
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 356, in visit
    new_node = super().visit(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 407, in visit
    return visitor(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 430, in visit_Subscript
    [self.visit(item) for item in slice_value.elts],
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 430, in <listcomp>
    [self.visit(item) for item in slice_value.elts],
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 356, in visit
    new_node = super().visit(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 407, in visit
    return visitor(node)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/setuptools/_vendor/typeguard/_transformer.py", line 489, in visit_Constant
    expression = ast.parse(node.value, mode="eval")
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/ast.py", line 50, in parse
    return compile(source, filename, mode, flags,
  File "<unknown>", line 1
    1 3 height width
      ^
SyntaxError: invalid syntax
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ pip list | grep ast
asttokens                 2.4.1
fastjsonschema            2.20.0
nvdiffrast                0.3.3
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ pip list | grep setuptools
setuptools                75.1.0
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ pip list | grep jaxtyping\
> ^C
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ pip list | grep jaxtyping
jaxtyping                 0.2.36
(textdeformer_new2) i-caiweiwei@caiweiwei-pc85d-564305-worker-0:/data/caiweiwei/TextDeformer-main$ pip install jaxtyping==0.2.34
Looking in indexes: http://mirrors.i.basemind.com/pypi/simple/, http://pypi.i.basemind.com/brain/dev/+simple


报错信息：cannot import name 'cached_download' from 'huggingface_hub'
Downgrading from 0.26.1 to 0.23.2 worked for me.