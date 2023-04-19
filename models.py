import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros, constant
from typing import Any, Callable, Sequence, Tuple
from functools import partial
import einops

ModuleDef = Any


class CNN(nn.Module):
    '''Classic Convolution Neural Network (CNN) with All-Convolution Architecture.'''

    num_filters: Sequence[int]
    num_strides: Sequence[int]
    mesh_vertexes: int
    pca_coef: jnp.array
    dtype: Any = jnp.float16
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, training: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype, kernel_init=nn.initializers.lecun_uniform())

        for i, filters in enumerate(self.num_filters):
            x = conv(filters, (3, 3), self.num_strides[i], name=f'conv{i}')(x)
            x = nn.relu(x)

        
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        # Set the dropout layer with a `rate` of 20%.
        # When the `deterministic` flag is `True`, dropout is turned off.
        x = nn.Dropout(rate=0.2)(x, deterministic=not training)
        
        x = nn.Dense(160)(x)
        x = nn.Dense(self.mesh_vertexes,
                     kernel_init=constant(jnp.array(self.pca_coef)))(x)
        x = einops.rearrange(x, 'b (n c) -> b n c', n=7306, c=3)

        return x



Classic_CNN = partial(
    CNN,
    num_filters=[64, 64, 96, 96, 144, 144, 216, 216, 324, 324, 486, 486],
    num_strides=[(2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2),
                 (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)])



        # x = jnp.reshape(x, (7306, 3))
        # x = nn.Dense(self.mesh_vertexes)(x)
        # x_dim = nn.Dense(self.mesh_vertexes)(x)
        # x_dim = nn.relu(x_dim)
        # x_dim = nn.Dense(self.mesh_vertexes)(x_dim)
        # x_dim = nn.relu(x_dim)
        # x_dim = nn.Dense(self.mesh_vertexes)(x_dim)

        # y_dim = nn.Dense(self.mesh_vertexes)(x)
        # y_dim = nn.relu(y_dim)
        # y_dim = nn.Dense(self.mesh_vertexes)(y_dim)
        # y_dim = nn.relu(y_dim)
        # y_dim = nn.Dense(self.mesh_vertexes)(y_dim)

        # z_dim = nn.Dense(self.mesh_vertexes)(x)
        # z_dim = nn.relu(z_dim)
        # z_dim = nn.Dense(self.mesh_vertexes)(z_dim)
        # z_dim = nn.relu(z_dim)
        # z_dim = nn.Dense(self.mesh_vertexes)(z_dim)

        # concat = jnp.stack([x_dim, y_dim, z_dim], axis=-1)
        # return concat
        # return jnp.reshape(concat, (jax.local_device_count(), -1)+ concat.shape[1:])
        # return einops.rearrange(concat, '(p b) d c -> p b d c')