from munch import Munch

from flax import linen as nn


class LeNet(nn.Module):
    @nn.compact
    def __call__(self, x, training:bool=True):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding=0)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding=0)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=0.25, deterministic=not training)(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=10)(x)
        return x
    
    
BENCHMARK2MODEL = {
    'MNIST': LeNet
}


def get_model(config:Munch) -> nn.Module:
    benchmark = config.default.benchmark
    if config.default.benchmark in BENCHMARK2MODEL.keys():
        model_class = BENCHMARK2MODEL[benchmark]
        return model_class()
    else:
        raise ValueError(f'No implemented model for {benchmark}')