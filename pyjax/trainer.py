from abc import ABC, abstractmethod
from munch import Munch
import jax
import jax.numpy as jnp  # JAX NumPy
import torch

from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers

from flax import linen as nn  # Linen API


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: metrics.Collection
    dropout_key: jax.random.PRNGKey


@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params, dropout_key):
        logits = state.apply_fn({'params': params}, 
                                batch['image'],
                                training=True,
                                rngs={'dropout': dropout_key}
                               )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params, state.dropout_key)
    state = state.apply_gradients(grads=grads)
    dropout_key = jax.random.split(key=state.dropout_key, num=2)[0]
    state = state.replace(dropout_key=dropout_key)
    return state


@jax.jit
def compute_metrics(state, batch):
    logits = state.apply_fn({'params': state.params}, 
                            batch['image'],
                            training=False
                           )
    loss = optax.softmax_cross_entropy_with_integer_labels(
               logits=logits, labels=batch['label']
           ).mean()
    metric_updates = state.metrics.single_from_model_output(
                         logits=logits, labels=batch['label'], loss=loss
                     )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


class ClassificationTrainer(ABC):
    def __init__(self, model: nn.Module, 
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader, 
                 config:Munch,
                ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.rng = jax.random.PRNGKey(config.default.seed)
        self.metrics = Metrics.empty()
        self.learning_rate = config.default.lr
        self.momentum = config.default.momentum
        self.input_size = config.default.input_size
        self.config = config

        self.create_train_state()
        self.metrics_history = {'train_loss': [],
                                'train_accuracy': [],
                                'test_loss': [],
                                'test_accuracy': []
                               }

    def create_train_state(self)  -> None:
        _, params_key, dropout_key = jax.random.split(key=self.rng, num=3)
        params = self.model.init({'params': params_key,
                                  'dropout': dropout_key
                                 }, 
                                 jnp.ones(self.input_size)
                                )['params']
        tx = optax.sgd(self.learning_rate, self.momentum)

        self.train_state =  TrainState.create(apply_fn=self.model.apply,
                                              params=params, tx=tx,
                                              metrics=self.metrics, 
                                              dropout_key=dropout_key
                                             )

    def log(self, i:int) -> None:
        print(f"train epoch: {i}, "
              f"loss: {self.metrics_history['train_loss'][-1]}, "
              f"accuracy: {self.metrics_history['train_accuracy'][-1] * 100}"
             )
        print(f"test epoch: {i}, "
              f"loss: {self.metrics_history['test_loss'][-1]}, "
              f"accuracy: {self.metrics_history['test_accuracy'][-1] * 100}"
             )

    def train(self) -> None:
        for i in range(self.config.default.epochs):
            """Train for a single epoch."""
            for image, label in self.train_loader:
                image = jnp.array(image).transpose((0, 2, 3, 1))
                label = jnp.array(label)
                batch = {'image': image, 'label': label}
                self.train_state = train_step(self.train_state, batch)
                self.train_state = compute_metrics(self.train_state, batch)

            for metric,value in self.train_state.metrics.compute().items():
                self.metrics_history[f'train_{metric}'].append(value)

            self.train_state = self.train_state.replace(
                                   metrics= self.train_state.metrics.empty()
                               )
            
            test_state = self.train_state
            for image, label in self.test_loader:
                image = jnp.array(image).transpose((0, 2, 3, 1))
                label = jnp.array(label)
                test_batch = {'image': image, 'label': label}
                test_state = compute_metrics(state=test_state,
                                             batch=test_batch
                                            )

            for metric,value in test_state.metrics.compute().items():
                self.metrics_history[f'test_{metric}'].append(value)
            
            self.log(i)

