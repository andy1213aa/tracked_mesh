import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Sequence, Tuple
from functools import partial

ModuleDef = Any


class CNN(nn.Module):
    '''Classic Convolution Neural Network (CNN) with All-Convolution Architecture.'''

    num_filters: Sequence[int]
    num_strides: Sequence[int]
    mesh_vertexes: int
    dtype: Any = jnp.float16
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)

        for i, filters in enumerate(self.num_filters):
            x = conv(filters, (3, 3), self.num_strides[i], name=f'conv{i}')(x)
            x = nn.relu(x)

        # x = nn.Dropout(rate=0.2)(x, deterministic=not train)
        x = nn.Dense(160)(x)
        x = nn.Dense(self.mesh_vertexes)(x)
        
        return x


Classic_CNN = partial(
    CNN,
    num_filters=[64, 64, 96, 96, 144, 144, 216, 216, 324, 324, 486, 486],
    num_strides=[(2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2),
                 (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)])


