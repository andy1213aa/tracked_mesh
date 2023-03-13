import ml_collections
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib

from jax import random as jrand
from typing import Any




class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale

def train_and_evalutation(config: ml_collections.ConfigDict) -> TrainState:
    """ Execute model training and evaluation loop."""

    rng = jrand.PRNGKey(0)

    param = 

