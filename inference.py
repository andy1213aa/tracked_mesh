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
    with open(pca_pth, 'rb') as f:
        pca = pickle.load(f)
    return pca.components_


def inititalized(key, render_size, model):
    input_shape = (1, render_size[0], render_size[1], 1)

    @jax.jit
    def init(*args):
        return model.init(*args, training=False)

    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))

    return variables


def create_model(model_cls, config, pca_coef, **kwargs):

    return model_cls(mesh_vertexes=config.vertex_num,
                     dtype=jnp.float32,
                     pca_coef=pca_coef)


class TrainState(train_state.TrainState):
    # batch_stats: Any
    key: jax.random.KeyArray
    dynamic_scale: dynamic_scale_lib.DynamicScale


def create_train_state(params_key, dropout_key,
                       config: ml_collections.ConfigDict, model, render_size,
                       learning_rate_fn):
    """Create initital training state."""

    variables = inititalized(params_key, render_size, model)
    tx = optax.adam(learning_rate=learning_rate_fn)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        #   batch_stats=batch_stats,
        key=dropout_key,
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

    root_key = jrand.PRNGKey(0)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
    steps_per_epoch = 1973

    model_cls = getattr(models, config.model)
    model = create_model(model_cls, config, get_pca_coef(config.pca))

    base_learning_rate = config.learning_rate * config.batch_size

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate,
                                               steps_per_epoch)

    state = create_train_state(params_key, dropout_key, config, model,
                               config.render_size, learning_rate_fn)

    state = restore_checkpoint(state, workdir)

    subject = '6674443'
    facial = 'E045_Jaw_Back'#'E001_Neutral_Eyes_Open'
    view = '400002'
    idx = '021885'#'000220'

    img = cv2.imread(
        f'/home/aaron/Desktop/multiface/{subject}_GHS/images/{facial}/{view}/{idx}.png'
    )

    img = cv2.resize(img, config.render_size)
    img = cv2.cvtColor(img,
                       cv2.COLOR_BGR2GRAY).reshape(config.render_size + (1, ))
    img = img.reshape((-1, ) + img.shape)
    img = jnp.asarray(img).astype(jnp.float16)
    image_mean = 47.727367
    image_std = 27.568207

    img = (img - image_mean) / image_std

    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    pred = state.apply_fn({'params': state.params},
                          img,
                          training=False,
                          rngs={'dropout': dropout_train_key})

    mean_mesh_pth = '/home/aaron/Desktop/multiface/6674443_GHS/geom/vert_mean.bin'
    with open(mean_mesh_pth, 'rb') as f:
        data = f.read()
        mesh_mean = np.frombuffer(data, dtype=np.float32).reshape((7306, 3))
        center = mesh_mean.mean(0)

    SCALE = np.max(np.abs(mesh_mean - center))

    pred_cpu = jax.device_get(pred)
    pred_cpu = einops.rearrange(pred_cpu, 'b v c -> (b v) c', c=3)  #b = 1
    pred_cpu = pred_cpu.copy()
    pred_cpu *= SCALE

    res, vertex_true = load_obj(
        f'/home/aaron/Desktop/multiface/{subject}_GHS/geom/tracked_mesh/{facial}/{idx}.obj'
    )

    with open(
            Path(
                f'/home/aaron/Desktop/multiface/{subject}_GHS/geom/tracked_mesh/{facial}/{idx}.bin'
            ),
            'rb',
    ) as f:
        data = f.read()
    vertex_true = np.frombuffer(data, dtype=np.float32).reshape((-1, 3))

    # vertex_true = np.array(vertex_true)

    print((np.mean((vertex_true - pred_cpu)**2)))

    obj_v_result = [f'v {x} {y} {z}\n' for x, y, z in pred_cpu]
    obj_v_result = np.array(obj_v_result)

    total_lines = 0
    with open(
            f'/home/aaron/Desktop/multiface/{subject}_GHS/geom/tracked_mesh/{facial}/{idx}.obj',
            'r') as f:
        total_lines = len(f.readlines())

    print(total_lines)
    txt = ''
    with open(
            f'/home/aaron/Desktop/multiface/{subject}_GHS/geom/tracked_mesh/{facial}/{idx}.obj',
            'r') as f:
        for i in range(7306):
            f.readline()
            txt += f'v {pred_cpu[i][0]} {pred_cpu[i][1]} {pred_cpu[i][2]}\n'
        for _ in range(7306, total_lines):

            txt += f.readline()

    with open('../test_data/test.obj', 'w') as w:
        w.write(txt)
    # with open(f'{idx}.obj', 'w') as f:
    #     f.writelines(obj_v_result)
