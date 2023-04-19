from absl import logging

import ml_collections
from clu import metric_writers
from clu import periodic_actions
import einops
from flax import jax_utils
from flax.training import checkpoints
from flax.training import train_state
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib

from functools import partial

from input_pipline import readTFRECORD

import jax
from jax import random as jrand
from jax import numpy as jnp
from jax import lax

import time
from typing import Any
import optax
import models


def get_pca_coef(pca_pth):
    import pickle
    with open(pca_pth, 'rb') as f:
        pca = pickle.load(f)
    return pca.components_


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


def inititalized(key, image_size, model):
    input_shape = (1, image_size[0], image_size[1], 3)

    @jax.jit
    def init(*args):
        return model.init(*args, training=False)

    # Dropout is disabled with `training=False` (that is, `deterministic=True`).
    variables = init({'params': key},
                     jnp.ones(input_shape, model.dtype)
                     )

    return variables


def mean_square_error_loss(pred, gt):

    return jnp.mean(jnp.square(jnp.subtract(pred, gt)))


def chamfer_distance_loss(pred_points, gt_points):
    diff = jnp.expand_dims(pred_points, axis=1) - jnp.expand_dims(gt_points,
                                                                  axis=0)
    distance = jnp.sum(jnp.square(jnp.linalg.norm(diff, axis=-1)), axis=-1)
    pred_to_gt = jnp.min(distance, axis=1)
    gt_to_pred = jnp.min(distance, axis=0)
    loss = jnp.mean(pred_to_gt) + jnp.mean(gt_to_pred)
    return loss


def earth_mover_distance_loss(pred_points, gt_points):
    distance_matrix = jnp.sqrt(
        jnp.sum(jnp.square(
            jnp.expand_dims(pred_points, axis=1) -
            jnp.expand_dims(gt_points, axis=0)),
                axis=-1))
    assignment_matrix = jax.linear_assignment(distance_matrix)
    assigned_distance = jnp.sum(distance_matrix[assignment_matrix[0],
                                                assignment_matrix[1]])
    return assigned_distance


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        # orbax_checkpointer.save(workdir, state, save_args=)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def create_input_iter(ds):
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


def compute_metrics(y_pred, y_true):
    loss = mean_square_error_loss(y_pred, y_true)
    # loss = chamfer_distance_loss(y_pred, y_true)

    metrics = {
        'loss': loss,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


class TrainState(train_state.TrainState):
    # batch_stats: Any
    key: jax.random.KeyArray
    dynamic_scale: dynamic_scale_lib.DynamicScale


def create_model(model_cls, config, pca_coef, **kwargs):

    return model_cls(mesh_vertexes=config.vertex,
                     dtype=jnp.float32,
                     pca_coef=pca_coef)


def create_learning_rate_fn(config: ml_collections.ConfigDict,
                            base_learning_rate: float, steps_per_epoch: int):

    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=config.warmup_epochs *
                                      steps_per_epoch)
    #inverse square root decay
    # ISR_fn = optax.constant_schedule(
        
    # )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    consine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                             decay_steps=cosine_epochs *
                                             steps_per_epoch)
    
    # optax.exponential_decay(init_value=consine_fn,decay_rate=)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, consine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])

    return schedule_fn
    # return warmup_fn


def create_train_state(params_key, dropout_key, config: ml_collections.ConfigDict, model,
                       image_size, learning_rate_fn):
    """Create initital training state."""

    variables = inititalized(params_key, image_size, model)
    tx = optax.adam(learning_rate=learning_rate_fn, b1=0.9, b2=0.999)
    # tx = optax.tx = optax.rmsprop(learning_rate=learning_rate_fn, eps=1e-4)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        key=dropout_key,
        #   batch_stats=batch_stats,
        dynamic_scale=None)
    return state


def train_step(state, batch, learning_rate_fn, dropout_key):

    def loss_fn(params):
        """loss function used for training."""
        # Dropout is enabled with `training=True` (that is, `deterministic=False`).
        pred = state.apply_fn({'params': params},
                              x=batch['img'],
                              training=True,
                              rngs={'dropout': dropout_train_key})
        
        loss = mean_square_error_loss(pred, batch['vtx'])
        return loss, pred

    step = state.step
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=step)
    # dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
    y_pred = aux[1]
    metrics = compute_metrics(y_pred, batch['vtx'])
    metrics['learning_rate'] = lr
    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics


def train_and_evalutation(config: ml_collections.ConfigDict, workdir: str,
                          datadir: str) -> TrainState:
    """ Execute model training and evaluation loop."""

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)

    root_key = jrand.PRNGKey(0)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
    ds = readTFRECORD(datadir, config)
    steps_per_epoch = 11
    steps_per_checkpoint = steps_per_epoch * 10

    train_iter = create_input_iter(ds)

    if config.num_train_steps <= 0:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    model_cls = getattr(models, config.model)

    model = create_model(model_cls, config, get_pca_coef(config.pca))

    base_learning_rate = config.learning_rate

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate,
                                               steps_per_epoch)

    state = create_train_state(params_key, dropout_key,config, model, config.image_size,
                               learning_rate_fn)

    state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(partial(train_step,
                                    learning_rate_fn=learning_rate_fn,
                                    dropout_key=dropout_key),
                            axis_name='batch')

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [
            periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
        ]

    train_metrics_last_t = time.time()

    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_steps), train_iter):

        state, metrics = p_train_step(state, batch)

        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')

        if config.get('log_every_steps'):
            train_metrics.append(metrics)

            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f'train_{k}': v
                    for k, v in jax.tree_util.tree_map(lambda x: x.mean(),
                                                       train_metrics).items()
                }
                summary['steps_per_second'] = config.log_every_steps / (
                    time.time() - train_metrics_last_t)

                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            logging.info('model save')
            save_checkpoint(state, workdir)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state
