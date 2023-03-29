import jax
import jax.numpy as jnp
import jax.random as jrand
from flax import linen as nn
from jax.nn.initializers import zeros, constant
from typing import Any, Callable, Sequence, Tuple
from functools import partial
import einops

ModuleDef = Any


# class PCACOEF_Initializer(nn.initializers.Initializer):

#     def __init__(self, my_array: jnp.ndarray):
#         self.my_array = my_array

#     def __call__(self, shape: Tuple[int, ...], dtype: Any, key: Any, *args,
#                  **kwargs):
        
#         return self.my_array


class CNN(nn.Module):
    '''Classic Convolution Neural Network (CNN) with All-Convolution Architecture.'''

    num_filters: Sequence[int]
    num_strides: Sequence[int]
    mesh_vertexes: int
    pca_coef: jnp.array
    dtype: Any = jnp.float16
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
       
        # zeros_init = jax.nn.initializers.zeros(rng, (160, 7306*3))
        # pcacoef_init = zeros_init + self.pca_coef
        for i, filters in enumerate(self.num_filters):
            x = conv(filters, (3, 3), self.num_strides[i], name=f'conv{i}')(x)
            x = nn.relu(x)

        # x = nn.Dropout(rate=0.2)(x, deterministic=not train)
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dense(160)(x)
        # x = nn.Dense(self.mesh_vertexes)(x)
        x = nn.Dense(self.mesh_vertexes,
                     kernel_init=constant(jnp.array(
                         self.pca_coef)))(x)

        # x_dim = nn.Dense(self.mesh_vertexes)(x)
        # y_dim = nn.Dense(self.mesh_vertexes)(x)
        # z_dim = nn.Dense(self.mesh_vertexes)(x)

        # concat = jnp.stack([x_dim, y_dim, z_dim], axis=-1)
        return x
        # return jnp.reshape(concat, (jax.local_device_count(), -1)+ concat.shape[1:])
        # return einops.rearrange(concat, '(p b) d c -> p b d c')


Classic_CNN = partial(
    CNN,
    num_filters=[64, 64, 96, 96, 144, 144, 216, 216, 324, 324, 486, 486],
    num_strides=[(2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2),
                 (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)])