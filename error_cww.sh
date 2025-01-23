#当修改输入的obj，没有重新换新的输出路径会报错（换个输出路径就可以啦）：
Traceback (most recent call last):
  File "/data/caiweiwei/TextDeformer-main/main.py", line 99, in <module>
    main()
  File "/data/caiweiwei/TextDeformer-main/main.py", line 92, in main
    loop(cfg)
  File "/data/caiweiwei/TextDeformer-main/loop.py", line 142, in loop
    gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0)) #load_mesh.v_pos.shape=torch.Size([1584, 3])  torch.Size([1, 398, 3])的时候报错
  File "/data/caiweiwei/TextDeformer-main/NeuralJacobianFields/SourceMesh.py", line 129, in jacobians_from_vertices
    return self.poisson.jacobians_from_vertices(v)
  File "/data/caiweiwei/TextDeformer-main/NeuralJacobianFields/PoissonSystem.py", line 311, in jacobians_from_vertices
    res = _multiply_sparse_2d_by_dense_3d(self.sparse_grad, V).type_as(V)
  File "/data/caiweiwei/TextDeformer-main/NeuralJacobianFields/PoissonSystem.py", line 632, in _multiply_sparse_2d_by_dense_3d
    C = mat.multiply_with_dense(B[i, ...])
  File "/data/caiweiwei/TextDeformer-main/NeuralJacobianFields/PoissonSystem.py", line 109, in multiply_with_dense
    res = torch_sparse.spmm(self.inds,self.vals, self.n, self.m, dense)
  File "/data/anconda3/envs/textdeformer_new2/lib/python3.9/site-packages/torch_sparse/spmm.py", line 22, in spmm
    assert n == matrix.size(-2)
AssertionError