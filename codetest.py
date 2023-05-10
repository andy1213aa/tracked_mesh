import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from clu import parameter_overview
import torch
import ffmpeg
import numpy as np
import torch as th
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
)
import jax.dlpack as jdlpack
import torch.utils.dlpack
import os
import time
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


np_array = jnp.ones((64, 1024, 1024, 3))
dlpack_array = jdlpack.to_dlpack(np_array)
th_dlpcak = torch.from_dlpack(dlpack_array)


np_array = jnp.ones((64, 1024, 1024, 3))
start = time.time()
dlpack_array = jdlpack.to_dlpack(np_array)
th_dlpcak = torch.from_dlpack(dlpack_array)
print(f'Time: {time.time() - start:.6f}s')


np_array = jnp.ones((64, 1024, 1024, 3))
tensor = torch.from_numpy(jax.device_get(np_array)).cuda()

np_array = jnp.ones((64, 1024, 1024, 3))
start = time.time()
tensor = torch.from_numpy(jax.device_get(np_array)).cuda()

print(f'Time: {time.time() - start:.6f}s')


# print(type(np_array))
# torch_ten = torch.from_numpy(np_array).cuda()

# # 创建一个 PyTorch 张量并将其移动到 CUDA 设备上
# x_torch = torch.randn(3, 4).cuda()

# # 将 PyTorch 张量转换为 JAX 数组
# x_jax = jax.device_put(x_torch)

# # 打印 JAX 数组的类型、形状和数据类型
# print(type(x_jax))  # <class 'jax.interpreters.xla.DeviceArray'>
# print(x_jax.shape)  # (3, 4)
# print(x_jax.dtype)  # float32


# # 创建一个 PyTorch 张量并将其移动到 CUDA 设备上
# x_torch = torch.randn(3, 4).cuda()

# # 将 PyTorch 张量转换为 NumPy 数组
# x_numpy = x_torch.cpu().detach().numpy()

# # 将 NumPy 数组转换为 JAX 数组
# x_jax = jnp.array(x_numpy)

# # 打印 JAX 数组的类型、形状和数据类型
# print(type(x_jax))  # <class 'jax.interpreters.xla.DeviceArray'>
# print(x_jax.shape)  # (3, 4)
# print(x_jax.dtype)  # float32