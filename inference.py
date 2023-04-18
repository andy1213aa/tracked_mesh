import jax
import jax.numpy as jnp
from jax import random as jrand

from flax.training import checkpoints
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib
import ml_collections

import cv2
import models
import os
from absl import logging
import einops
import optax
from pathlib import Path
import numpy as np


def load_obj(pth):

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    obj_pth = Path(pth)

    if not obj_pth.exists:
        logging.info(f'file {obj_pth} does not exist')
        return None, False
    else:
        with open(obj_pth, 'r') as f:
            vertices = []
            for line in f:
                if 'v' in line and 'vn' not in line and 'vt' not in line:

                    v = [
                        float(x) for x in list(
                            filter(lambda x: is_number(x), line.split()))
                    ]
                    vertices.append(v)
                if 'f' in line:
                    break
            return True, vertices


def get_pca_coef(pca_pth):
    import pickle
    with open('pca.pickle', 'rb') as f:
        pca = pickle.load(f)
    return pca.components_


def inititalized(key, image_size, model):
    input_shape = (1, image_size[0], image_size[1], 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))

    return variables


def create_model(model_cls, config, pca_coef, **kwargs):

    return model_cls(mesh_vertexes=config.vertex,
                     dtype=jnp.float32,
                     pca_coef=pca_coef)


class TrainState(train_state.TrainState):
    # batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


def create_train_state(rng, config: ml_collections.ConfigDict, model,
                       image_size, learning_rate_fn):
    """Create initital training state."""

    variables = inititalized(rng, image_size, model)
    tx = optax.adam(learning_rate=learning_rate_fn)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        #   batch_stats=batch_stats,
        dynamic_scale=None)
    return state


def create_learning_rate_fn(config: ml_collections.ConfigDict,
                            base_learning_rate: float, steps_per_epoch: int):

    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=config.batch_size *
                                      steps_per_epoch)

    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    consine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                             decay_steps=cosine_epochs *
                                             steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, consine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])

    return schedule_fn


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def inference(
    config: ml_collections.ConfigDict,
    workdir: str,
):

    rng = jrand.PRNGKey(0)
    steps_per_epoch = 1973

    model_cls = getattr(models, config.model)
    model = create_model(model_cls, config, get_pca_coef(config.pca))

    base_learning_rate = config.learning_rate * config.batch_size

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate,
                                               steps_per_epoch)

    state = create_train_state(rng, config, model, config.image_size,
                               learning_rate_fn)

    state = restore_checkpoint(state, workdir)

    idx = '002067'
    img = cv2.imread(
        f'/home/aaron/Desktop/multiface/6674443_GHS/images/E006_Jaw_Drop_Brows_Up/400016/{idx}.png'
    )
    cv2.imwrite('019278.png', img)
    img = cv2.resize(img, config.image_size)
    img = img.reshape((-1, ) + img.shape)
    img = jnp.asarray(img).astype(jnp.float16)

    pred = state.apply_fn({'params': state.params}, img)
    pred_cpu = jax.device_get(pred)
    pred_cpu = einops.rearrange(pred_cpu, 'b (v c) -> (b v) c', c=3)

    res, vertex_true = load_obj(
        f'/home/aaron/Desktop/multiface/6674443_GHS/geom/tracked_mesh/E006_Jaw_Drop_Brows_Up/{idx}.obj'
    )
    vertex_true = np.array(vertex_true)

    print((np.mean(vertex_true - pred_cpu)**2))

    obj_v_result = [f'v {x} {y} {z}\n' for x, y, z in pred_cpu]
    obj_v_result = np.array(obj_v_result)
    with open(f'{idx}.obj', 'w') as f:
        f.writelines(obj_v_result)
