import jax
import jax.numpy as jnp
from jax import random as jrand

from flax.training import checkpoints
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib
import ml_collections
from input_pipline import readTFRECORD
import cv2
import models
import os
from absl import logging
import einops
import optax
from pathlib import Path
import numpy as np
import pickle
from flax import jax_utils
import time


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

    with open(pca_pth, 'rb') as f:
        pca = pickle.load(f)
    return pca.components_


def get_mean_mesh(mean_mesh_pth):

    with open(mean_mesh_pth, 'rb') as f:
        mean_mesh = pickle.load(f)
    return mean_mesh


def inititalized(key, render_size, model):
    input_shape = (1, render_size[0], render_size[1], 1)

    @jax.jit
    def init(*args):
        return model.init(*args, training=False)

    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))

    return variables


def create_input_iter(ds):
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


def create_model(model_cls, config, pca_coef, **kwargs):

    return model_cls(
        mesh_vertexes=config.vertex_num,
        dtype=jnp.float32,
        pca_basis=pca_coef,
        mean_mesh=get_mean_mesh(config.mean_mesh),
    )


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)

        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


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

    # subject = '6674443'
    # facial = 'E061_Lips_Puffed'  #'E001_Neutral_Eyes_Open'
    # view = '400002'
    # idx = '029397'  #'000220'

    ds = readTFRECORD('../training_data/1015_18pose_664nums.tfrecord', config)
    train_iter = create_input_iter(ds)

    count = 0
    mse = 0
    ex_time = 0

    num_steps = int(config.steps_per_epoch * config.num_epochs)
    for step, batch in zip(range(0, num_steps), train_iter):

        ######
        img = cv2.imread(
            '/media/aaron/work/KIRI/back_up_scan/P09_EYE_OPEN_SMILE/images_rvm/0021.png'
        ).astype(np.float32)
        img = cv2.resize(img, (756, 1008))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img /= 255.
        ######

        ######
        # img = batch['img'][0][0]
        ######
        cv2.imwrite(f'test_{count}.png', jax.device_get(img) * 255)
        img = img.reshape((1, 1008, 756, 1))
        vtx = batch['vtx'][0][0].reshape((7306, 3))
        dropout_train_key = jax.random.fold_in(key=dropout_key,
                                               data=state.step)
        start = time.time()
        pred = state.apply_fn({'params': state.params},
                              img,
                              training=False,
                              rngs={'dropout': dropout_train_key})
        ex_t = time.time() - start
        pred_cpu = jax.device_get(pred)
        pred_cpu = einops.rearrange(pred_cpu, 'b (v c) -> (b v) c',
                                    c=3)  #b = 1
        pred_cpu = pred_cpu.copy()

        print(np.mean((vtx - pred_cpu)**2))
        print(f'EX_time: {ex_t:0.5f}')
        mse += np.mean((vtx - pred_cpu)**2)

        if count > 0:
            ex_time += ex_t

        obj_v_result = [f'v {x} {y} {z}\n' for x, y, z in pred_cpu]
        obj_v_result = np.array(obj_v_result)

        #讀進所有的v, vt, f
        total_lines = 0
        with open(
                f'/media/aaron/work/KIRI/ME_1015/P11_MOUTH_RIGHT/model_align.obj',
                'r') as f:
            total_lines = len(f.readlines())

        txt = ''
        #只改v，其他不因obj不同而改變
        with open(
                f'/media/aaron/work/KIRI/ME_1015/P11_MOUTH_RIGHT/model_align.obj',
                'r') as f:
            for i in range(7306):
                f.readline()
                txt += f'v {pred_cpu[i][0]} {pred_cpu[i][1]} {pred_cpu[i][2]}\n'
            for _ in range(7306, total_lines):

                txt += f.readline()
        # with open('../test_data/test.obj', 'w') as w:

        with open(f'test_{count}.obj', 'w') as w:
            w.write(txt)

        if count == 100:
            print(f'Average Mean-Square-Error: {mse/count:0.5f}mm.')
            print(f'Average Execution-Time: {ex_time/count:0.5f}s.')
            break
        count += 1

        break
