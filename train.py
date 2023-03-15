import ml_collections
from clu import metric_writers
from clu import periodic_actions

from flax.training import checkpoints
from flax.training import train_state
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib

from functools import partial
from absl import logging

import jax
from jax import random as jrand
from jax import numpy as jnp
from jax import lax

import time
from typing import Any
import optax
import models

VERTEXES = 3000
steps_per_epoch = 50


def inititalized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
    # logging.info(variables)
    # return variables['params'], variables['batch_stats']
    return variables


def mean_square_error_loss(pred, gt):
    return jnp.mean((pred - gt)**2)


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def compute_metrics(y_pred, y_true):
    loss = mean_square_error_loss(y_pred, y_true)
    metrics = {
        'loss': loss,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


class TrainState(train_state.TrainState):
    # batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


def create_model(model_cls, **kwargs):

    return model_cls(mesh_vertexes=VERTEXES, dtype=jnp.float32)


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


def train_step(state, batch, learning_rate_fn):

    def loss_fn():
        pass

    step = state.step
    # dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
    new_model_state, y_pred = aux[1]
    metrics = compute_metrics(y_pred, batch['y_truth'])
    metrics['learning_rate'] = lr
    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics


def train_and_evalutation(config: ml_collections.ConfigDict,
                          workdir: str) -> TrainState:
    """ Execute model training and evaluation loop."""

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)

    rng = jrand.PRNGKey(0)

    steps_per_checkpoint = steps_per_epoch * 10
    
    if config.num_train_steps <= 0:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps
    
    model_cls = getattr(models, config.model)
    model = create_model(model_cls)

    base_learning_rate = config.learning_rate * config.batch_size / 256.

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate,
                                               steps_per_epoch)

    state = create_train_state(rng, config, model, config.image_size,
                               learning_rate_fn)

    state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    p_train_step = jax.pmap(partial(train_step,
                                    learning_rate_fn=learning_rate_fn),
                            axis_name='batch')
    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [
            periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
        ]

    train_metrics_last_t = time.time()

    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in dataset:
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
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir)
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state