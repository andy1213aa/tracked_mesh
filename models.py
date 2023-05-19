import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros, constant
from typing import Any, Callable, Sequence, Tuple
from functools import partial
import einops

ModuleDef = Any


class ExpressionEncoder(nn.Module):
    '''
    Encode 2D image
    '''
    num_filters: Sequence[int]
    num_strides: Sequence[int]
    code: int = 128
    dtype: Any = jnp.float32
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, training: bool = True):
        conv = partial(self.conv,
                       use_bias=False,
                       dtype=self.dtype,
                       kernel_init=nn.initializers.lecun_uniform())

        for i, filters in enumerate(self.num_filters):
            x = conv(filters, (3, 3), self.num_strides[i], name=f'conv{i}')(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dropout(rate=0.2)(x, deterministic=not training)
        x = nn.Dense(self.code)(x)

        return x


class VertexUNet(nn.Module):
    '''
    Main model bone.
    U-Net that inspired from meshtalk.
    
    return: (B, V, C) tensor
    '''

    enconding_units: Sequence[int]
    decoding_unints: Sequence[int]
    expr_filters: Sequence[int]
    expr_strides: Sequence[int]
    mesh_vertexes: int = 7306
    expr_code: int = 128
    dtype: Any = jnp.float32
    dense: ModuleDef = nn.Dense

    @nn.compact
    def __call__(self, x, img, training: bool = True):

        expr_encoder = ExpressionEncoder(num_filters=self.expr_filters,
                                         num_strides=self.expr_strides,
                                         code=self.expr_code)
        # print(type(self.decoding_unints))
        # self.decoding_unints[-1] = self.mesh_vertexes * 3
        dense = partial(
            self.dense,
            use_bias=True,
            dtype=self.dtype,
        )
        skips = []

        #expression encoder
        expr_encoding = expr_encoder(img, training=training)

        #Encoding
        for i, units in enumerate(self.enconding_units):
            skips = [x] + skips

            x = dense(units, name=f'Encoding_Dense_{i}')(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = jnp.concatenate([x, expr_encoding], axis=-1)

        #Decoding
        for i, units in enumerate(self.decoding_unints):
            x = dense(units, name=f'Decoding_Dense_{i}')(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            x = skips[i] + x

        # x = einops.rearrange(x, 'b (v c) -> b v c', v=self.mesh_vertexes, c=3)

        return x


class CNN(nn.Module):
    '''Classic Convolution Neural Network (CNN) with All-Convolution Architecture.'''

    num_filters: Sequence[int]
    num_strides: Sequence[int]
    mesh_vertexes: int
    pca_coef: jnp.array
    dtype: Any = jnp.float32
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, training: bool = True):
        conv = partial(self.conv,
                       use_bias=False,
                       dtype=self.dtype,
                       kernel_init=nn.initializers.lecun_uniform())

        for i, filters in enumerate(self.num_filters):
            x = conv(filters, (3, 3), self.num_strides[i], name=f'conv{i}')(x)
            x = nn.relu(x)

        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        # Set the dropout layer with a `rate` of 20%.
        # When the `deterministic` flag is `True`, dropout is turned off.
        x = nn.Dropout(rate=0.2)(x, deterministic=not training)

        x = nn.Dense(160)(x)
        x = nn.Dense(self.mesh_vertexes * 3,
                     # kernel_init=constant(jnp.array(self.pca_coef)),
                     )(x)
        x = einops.rearrange(x, 'b (n c) -> b n c', n=7306, c=3)
        return x


Classic_CNN = partial(
    CNN,
    num_filters=[64, 64, 96, 96, 144, 144, 216, 216, 324, 324, 486, 486],
    num_strides=[(2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2),
                 (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)])

# Classic_UNet = partial(VertexUNet,
#                        enconding_units=[512, 256, 128],
#                        decoding_unints=[256, 512, 7306 * 3],
#                        expr_filters=[16, 16, 32, 32, 64, 64, 128, 128],
#                        expr_strides=[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1),
#                                      (2, 2), (1, 1), (2, 2)],
#                        expr_code=128)
